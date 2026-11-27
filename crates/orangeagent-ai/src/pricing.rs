use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{AiError, ProviderFamily, TokenUsage};

/// Canonical public endpoint for the community-maintained `models.dev` catalog.
pub const MODELS_DEV_API_URL: &str = "https://models.dev/api.json";

/// Compile-time embedded baseline snapshot in `models.dev/api.json` schema.
pub const EMBEDDED_MODELS_DEV_SNAPSHOT: &str = include_str!("../data/models.dev.snapshot.json");

/// Context window limits, capabilities, and per-million-token USD pricing for a model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelPricing {
    #[serde(default)]
    pub provider_id: String,
    pub model: String,
    pub family: ProviderFamily,
    pub context_window: u64,
    pub max_output_tokens: u32,
    pub input_per_million_usd: f64,
    pub output_per_million_usd: f64,
    #[serde(default)]
    pub cache_read_per_million_usd: f64,
    #[serde(default)]
    pub cache_write_per_million_usd: f64,
    #[serde(default)]
    pub supports_tool_call: bool,
    #[serde(default)]
    pub supports_reasoning: bool,
}

/// Itemized USD cost breakdown for one LLM turn or cumulative ledger.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct CostBreakdown {
    pub uncached_input_usd: f64,
    pub cache_read_usd: f64,
    pub cache_write_usd: f64,
    pub output_usd: f64,
    pub total_usd: f64,
    /// Estimated USD saved compared to paying full uncached input price for `cache_read_tokens`.
    pub cache_savings_usd: f64,
}

impl ModelPricing {
    /// Compute itemized USD cost for a normalized [`TokenUsage`].
    ///
    /// Follows `TokenUsage`'s invariant where `input_tokens` is the **total**
    /// prompt input (including `cache_read_tokens` and `cache_write_tokens`).
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_cost(&self, usage: &TokenUsage) -> CostBreakdown {
        let cached_total = usage
            .cache_read_tokens
            .saturating_add(usage.cache_write_tokens);
        let uncached_tokens = usage.input_tokens.saturating_sub(cached_total);

        let uncached_input_usd =
            (uncached_tokens as f64 / 1_000_000.0) * self.input_per_million_usd;
        let cache_read_usd =
            (usage.cache_read_tokens as f64 / 1_000_000.0) * self.cache_read_per_million_usd;
        let cache_write_usd =
            (usage.cache_write_tokens as f64 / 1_000_000.0) * self.cache_write_per_million_usd;
        let output_usd = (usage.output_tokens as f64 / 1_000_000.0) * self.output_per_million_usd;
        let total_usd = uncached_input_usd + cache_read_usd + cache_write_usd + output_usd;

        let cache_savings_usd = ((usage.cache_read_tokens as f64 / 1_000_000.0)
            * (self.input_per_million_usd - self.cache_read_per_million_usd))
            .max(0.0);

        CostBreakdown {
            uncached_input_usd,
            cache_read_usd,
            cache_write_usd,
            output_usd,
            total_usd,
            cache_savings_usd,
        }
    }
}

/// Multi-source model catalog supporting `models.dev` JSON, LiteLLM JSON,
/// compile-time embedded baselines, and runtime user overrides.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PricingCatalog {
    /// Entries keyed by `"{provider_id}/{model_id}"` (lowercased).
    qualified: BTreeMap<String, ModelPricing>,
    /// Entries keyed by bare `"{model_id}"` (lowercased), favoring first-party providers.
    bare: BTreeMap<String, ModelPricing>,
}

impl Default for PricingCatalog {
    fn default() -> Self {
        Self::builtin()
    }
}

impl PricingCatalog {
    /// Construct an empty catalog.
    pub fn empty() -> Self {
        Self {
            qualified: BTreeMap::new(),
            bare: BTreeMap::new(),
        }
    }

    /// Load the compile-time embedded `models.dev` snapshot (`EMBEDDED_MODELS_DEV_SNAPSHOT`).
    pub fn builtin() -> Self {
        Self::from_models_dev_json(EMBEDDED_MODELS_DEV_SNAPSHOT).unwrap_or_else(|_| Self::empty())
    }

    /// Parse a catalog in `https://models.dev/api.json` format.
    ///
    /// Incomplete entries (such as embedding models without `cost` or `limit.context`)
    /// are skipped gracefully. First-party providers take priority over aggregators
    /// for bare-name lookups while keeping all `provider/model` entries queryable.
    pub fn from_models_dev_json(raw: &str) -> Result<Self, AiError> {
        let providers: BTreeMap<String, ModelsDevProvider> = serde_json::from_str(raw)
            .map_err(|e| AiError::Protocol(format!("invalid models.dev catalog JSON: {e}")))?;

        let mut catalog = Self::empty();
        for (provider_id, provider) in providers {
            let provider_lower = provider_id.trim().to_ascii_lowercase();
            let npm = provider.npm.as_deref();
            for (model_id, spec) in provider.models {
                let Some(limit) = spec.limit else {
                    continue;
                };
                let Some(context_window) = limit.context.filter(|&c| c > 0) else {
                    continue;
                };
                let Some(cost) = spec.cost else {
                    continue;
                };
                let pricing = ModelPricing {
                    provider_id: provider_lower.clone(),
                    model: model_id.clone(),
                    family: infer_provider_family(&provider_lower, npm, &model_id),
                    context_window,
                    max_output_tokens: limit
                        .output
                        .and_then(|v| u32::try_from(v).ok())
                        .unwrap_or(8_192),
                    input_per_million_usd: cost.input.unwrap_or(0.0),
                    output_per_million_usd: cost.output.unwrap_or(0.0),
                    cache_read_per_million_usd: cost.cache_read.unwrap_or(0.0),
                    cache_write_per_million_usd: cost.cache_write.unwrap_or(0.0),
                    supports_tool_call: spec.tool_call,
                    supports_reasoning: spec.reasoning,
                };
                catalog.insert_with_priority(pricing);
            }
        }
        Ok(catalog)
    }

    /// Parse LiteLLM's `model_prices_and_context_window.json` format.
    ///
    /// Converts per-token USD rates (`* 1_000_000.0`) into per-million USD rates
    /// and skips `"sample_spec"` or entries without token costs.
    pub fn from_litellm_json(raw: &str) -> Result<Self, AiError> {
        let entries: BTreeMap<String, Value> = serde_json::from_str(raw)
            .map_err(|e| AiError::Protocol(format!("invalid LiteLLM pricing JSON: {e}")))?;

        let mut catalog = Self::empty();
        for (key, val) in entries {
            if key == "sample_spec" {
                continue;
            }
            let Some(pricing) = parse_litellm_entry(&key, &val) else {
                continue;
            };
            catalog.insert_with_priority(pricing);
        }
        Ok(catalog)
    }

    /// Assemble a layered catalog without doing network I/O on the lookup path:
    ///
    /// 1. Embedded baseline snapshot (`Self::builtin()`)
    /// 2. Optional caller-cached `models.dev` JSON (corrupted JSON is ignored gracefully)
    /// 3. User/config overrides (highest precedence)
    pub fn layered(
        cached_models_dev_json: Option<&str>,
        overrides: impl IntoIterator<Item = ModelPricing>,
    ) -> Self {
        let mut catalog = Self::builtin();
        if let Some(raw) = cached_models_dev_json
            && let Ok(cached) = Self::from_models_dev_json(raw)
        {
            catalog.merge_catalog(cached);
        }
        catalog.merge_overrides(overrides);
        catalog
    }

    /// Explicitly fetch and parse a remote `models.dev/api.json` catalog.
    ///
    /// Returns both the parsed [`PricingCatalog`] and the raw JSON string so callers
    /// (e.g. `cli`/`spine`) can atomically persist it to a local cache file.
    pub async fn fetch_models_dev(
        client: &reqwest::Client,
        url: &str,
    ) -> Result<(Self, String), AiError> {
        let response = client
            .get(url)
            .send()
            .await
            .map_err(|e| AiError::Network(format!("failed to fetch models catalog: {e}")))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(AiError::Http {
                status: status.as_u16(),
                message: body,
                retry_after_ms: None,
            });
        }
        let raw = response
            .text()
            .await
            .map_err(|e| AiError::Network(format!("failed to read models catalog body: {e}")))?;
        let catalog = Self::from_models_dev_json(&raw)?;
        Ok((catalog, raw))
    }

    /// Insert or unconditionally override a model entry (highest priority).
    pub fn register(&mut self, pricing: ModelPricing) {
        let model_key = bare_model_key(&pricing.model);
        if !pricing.provider_id.trim().is_empty() {
            let qualified_key = format!(
                "{}/{}",
                pricing.provider_id.trim().to_ascii_lowercase(),
                model_key
            );
            self.qualified.insert(qualified_key, pricing.clone());
        }
        self.bare.insert(model_key, pricing);
    }

    /// Apply a sequence of custom/config overrides onto `self`.
    pub fn merge_overrides(&mut self, overrides: impl IntoIterator<Item = ModelPricing>) {
        for item in overrides {
            self.register(item);
        }
    }

    /// Merge all entries from `higher_priority` on top of `self`.
    pub fn merge_catalog(&mut self, higher_priority: Self) {
        for (k, v) in higher_priority.qualified {
            self.qualified.insert(k, v);
        }
        for (k, v) in higher_priority.bare {
            self.bare.insert(k, v);
        }
    }

    /// Look up pricing by qualified `provider/model`, exact bare ID, or delimited
    /// longest-prefix match (e.g. `claude-sonnet-4-5-20250929` -> `claude-sonnet-4-5`).
    pub fn lookup(&self, model: &str) -> Option<&ModelPricing> {
        let normalized = model.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            return None;
        }

        if let Some((provider, bare_name)) = normalized.rsplit_once('/') {
            if let Some(exact) = self.qualified.get(&normalized) {
                return Some(exact);
            }
            if let Some(prefixed) =
                longest_delimited_match(&self.qualified, &normalized, Some(provider))
            {
                return Some(prefixed);
            }
            return self
                .bare
                .get(bare_name)
                .or_else(|| longest_delimited_match(&self.bare, bare_name, None));
        }

        self.bare
            .get(&normalized)
            .or_else(|| longest_delimited_match(&self.bare, &normalized, None))
    }

    /// Estimate USD cost for `model` if present in the catalog.
    pub fn estimate_cost(&self, model: &str, usage: &TokenUsage) -> Option<CostBreakdown> {
        self.lookup(model).map(|p| p.estimate_cost(usage))
    }

    /// Look up context window size (in tokens) for `model`.
    pub fn context_window(&self, model: &str) -> Option<u64> {
        self.lookup(model).map(|p| p.context_window)
    }

    /// Number of distinct bare model IDs registered in the catalog.
    pub fn len(&self) -> usize {
        self.bare.len()
    }

    /// Whether the catalog has zero models.
    pub fn is_empty(&self) -> bool {
        self.bare.is_empty()
    }

    fn insert_with_priority(&mut self, pricing: ModelPricing) {
        let model_key = bare_model_key(&pricing.model);
        if !pricing.provider_id.trim().is_empty() {
            let qualified_key = format!(
                "{}/{}",
                pricing.provider_id.trim().to_ascii_lowercase(),
                model_key
            );
            self.qualified.insert(qualified_key, pricing.clone());
        }

        let replace_bare = match self.bare.get(&model_key) {
            None => true,
            Some(existing) => {
                provider_priority(&pricing.provider_id) >= provider_priority(&existing.provider_id)
            }
        };
        if replace_bare {
            self.bare.insert(model_key, pricing);
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelsDevProvider {
    #[serde(default)]
    npm: Option<String>,
    #[serde(default)]
    models: BTreeMap<String, ModelsDevModel>,
}

#[derive(Debug, Deserialize)]
struct ModelsDevModel {
    #[serde(default)]
    reasoning: bool,
    #[serde(default)]
    tool_call: bool,
    #[serde(default)]
    limit: Option<ModelsDevLimit>,
    #[serde(default)]
    cost: Option<ModelsDevCost>,
}

#[derive(Debug, Deserialize)]
struct ModelsDevLimit {
    #[serde(default)]
    context: Option<u64>,
    #[serde(default)]
    output: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ModelsDevCost {
    #[serde(default)]
    input: Option<f64>,
    #[serde(default)]
    output: Option<f64>,
    #[serde(default)]
    cache_read: Option<f64>,
    #[serde(default)]
    cache_write: Option<f64>,
}

fn parse_litellm_entry(raw_key: &str, val: &Value) -> Option<ModelPricing> {
    let input_per_token = val.get("input_cost_per_token")?.as_f64()?;
    let output_per_token = val.get("output_cost_per_token")?.as_f64()?;
    let context_window = val
        .get("max_input_tokens")
        .or_else(|| val.get("max_tokens"))
        .and_then(Value::as_u64)
        .filter(|&n| n > 0)?;

    let max_output_tokens = val
        .get("max_output_tokens")
        .and_then(Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .unwrap_or(8_192);

    let cache_read_per_token = val
        .get("cache_read_input_token_cost")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let cache_write_per_token = val
        .get("cache_creation_input_token_cost")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);

    let litellm_provider = val
        .get("litellm_provider")
        .and_then(Value::as_str)
        .unwrap_or("");
    let (key_provider, bare) = raw_key
        .rsplit_once('/')
        .map_or((litellm_provider, raw_key), |(p, m)| (p, m));
    let provider_id = key_provider.trim().to_ascii_lowercase();

    Some(ModelPricing {
        provider_id: provider_id.clone(),
        model: bare.to_string(),
        family: infer_provider_family(&provider_id, None, bare),
        context_window,
        max_output_tokens,
        input_per_million_usd: input_per_token * 1_000_000.0,
        output_per_million_usd: output_per_token * 1_000_000.0,
        cache_read_per_million_usd: cache_read_per_token * 1_000_000.0,
        cache_write_per_million_usd: cache_write_per_token * 1_000_000.0,
        supports_tool_call: val
            .get("supports_function_calling")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        supports_reasoning: val
            .get("supports_reasoning")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn infer_provider_family(provider_id: &str, npm: Option<&str>, model_id: &str) -> ProviderFamily {
    let p = provider_id.to_ascii_lowercase();
    let m = model_id.to_ascii_lowercase();
    let npm_lower = npm.unwrap_or("").to_ascii_lowercase();

    if p == "anthropic" || npm_lower.contains("anthropic") {
        return ProviderFamily::Anthropic;
    }
    if p == "google" || p == "gemini" || npm_lower.contains("google") {
        return ProviderFamily::Gemini;
    }
    if p == "openai"
        && (m == "o1"
            || m.starts_with("o1-")
            || m == "o3"
            || m.starts_with("o3-")
            || m.starts_with("o4-"))
    {
        return ProviderFamily::OpenAiResponses;
    }
    ProviderFamily::OpenAiChat
}

fn provider_priority(provider_id: &str) -> u8 {
    match provider_id {
        "anthropic" | "openai" | "deepseek" | "google" => 3,
        "azure" | "amazon-bedrock" | "google-vertex" => 2,
        "" => 0,
        _ => 1,
    }
}

fn bare_model_key(model: &str) -> String {
    let trimmed = model.trim();
    let base = trimmed
        .rsplit_once('/')
        .map_or(trimmed, |(_, suffix)| suffix);
    base.to_ascii_lowercase()
}

fn longest_delimited_match<'a>(
    map: &'a BTreeMap<String, ModelPricing>,
    query: &str,
    required_provider: Option<&str>,
) -> Option<&'a ModelPricing> {
    let mut best: Option<(&str, &ModelPricing)> = None;
    for (key, pricing) in map {
        if let Some(req_p) = required_provider
            && pricing.provider_id != req_p
        {
            continue;
        }
        if is_delimited_prefix(query, key)
            && best.is_none_or(|(best_key, _)| key.len() > best_key.len())
        {
            best = Some((key.as_str(), pricing));
        }
    }
    best.map(|(_, pricing)| pricing)
}

fn is_delimited_prefix(query: &str, candidate: &str) -> bool {
    let Some(rest) = query.strip_prefix(candidate) else {
        return false;
    };
    matches!(rest.as_bytes().first(), Some(b'-' | b':' | b'@'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_snapshot_loads_and_resolves_exact_dated_and_prefixed_names() {
        let catalog = PricingCatalog::builtin();
        assert!(!catalog.is_empty());

        let exact = catalog
            .lookup("claude-sonnet-4-6")
            .expect("exact sonnet 4.6");
        assert_eq!(exact.context_window, 200_000);
        assert_eq!(exact.family, ProviderFamily::Anthropic);
        assert!(exact.supports_tool_call);
        assert!(exact.supports_reasoning);

        let dated = catalog
            .lookup("claude-sonnet-4-5-20250929")
            .expect("dated sonnet 4.5");
        assert_eq!(dated.model, "claude-sonnet-4-5");

        let prefixed = catalog
            .lookup("anthropic/claude-haiku-4-5-20251001")
            .expect("provider-prefixed haiku");
        assert_eq!(prefixed.model, "claude-haiku-4-5");

        let mini = catalog
            .lookup("gpt-4.1-mini-2025-04-14")
            .expect("longest prefix matches gpt-4.1-mini over gpt-4.1");
        assert_eq!(mini.model, "gpt-4.1-mini");
        assert_eq!(catalog.context_window("gpt-4.1"), Some(1_047_576));

        assert!(catalog.lookup("nonexistent-model-xyz").is_none());
    }

    #[test]
    fn models_dev_parser_prioritizes_first_party_and_skips_incomplete_entries() {
        let json = r#"{
          "openrouter": {
            "name": "OpenRouter",
            "npm": "@openrouter/ai-sdk-provider",
            "models": {
              "claude-sonnet-4-6": {
                "tool_call": true,
                "limit": { "context": 200000, "output": 64000 },
                "cost": { "input": 3.5, "output": 16.0 }
              },
              "text-embedding-3-large": {
                "limit": { "context": 8191 }
              }
            }
          },
          "anthropic": {
            "name": "Anthropic",
            "npm": "@ai-sdk/anthropic",
            "models": {
              "claude-sonnet-4-6": {
                "reasoning": true,
                "tool_call": true,
                "limit": { "context": 200000, "output": 64000 },
                "cost": { "input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75 }
              }
            }
          }
        }"#;

        let catalog = PricingCatalog::from_models_dev_json(json).expect("valid models.dev json");
        // Embedding model without `cost` was skipped without error
        assert_eq!(catalog.len(), 1);
        // Bare lookup picks direct Anthropic over OpenRouter regardless of BTreeMap key order
        let bare = catalog.lookup("claude-sonnet-4-6").expect("bare");
        assert_eq!(bare.provider_id, "anthropic");
        assert!((bare.input_per_million_usd - 3.0).abs() < 1e-9);
        // Qualified OpenRouter lookup remains accessible
        let or = catalog
            .lookup("openrouter/claude-sonnet-4-6")
            .expect("qualified openrouter");
        assert_eq!(or.provider_id, "openrouter");
        assert!((or.input_per_million_usd - 3.5).abs() < 1e-9);
    }

    #[test]
    fn litellm_json_parser_converts_per_token_rates_and_skips_sample_spec() {
        let litellm_json = r#"{
          "sample_spec": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0
          },
          "anthropic/claude-sonnet-4-6": {
            "max_input_tokens": 200000,
            "max_output_tokens": 64000,
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 15e-6,
            "cache_read_input_token_cost": 3e-7,
            "cache_creation_input_token_cost": 3.75e-6,
            "litellm_provider": "anthropic",
            "supports_function_calling": true,
            "supports_reasoning": true
          }
        }"#;

        let catalog = PricingCatalog::from_litellm_json(litellm_json).expect("litellm json");
        assert_eq!(catalog.len(), 1);
        let entry = catalog.lookup("claude-sonnet-4-6").expect("sonnet");
        assert_eq!(entry.family, ProviderFamily::Anthropic);
        assert!((entry.input_per_million_usd - 3.0).abs() < 1e-9);
        assert!((entry.output_per_million_usd - 15.0).abs() < 1e-9);
        assert!((entry.cache_read_per_million_usd - 0.3).abs() < 1e-9);
        assert!((entry.cache_write_per_million_usd - 3.75).abs() < 1e-9);
    }

    #[test]
    fn layered_catalog_applies_cached_snapshot_and_user_overrides_with_corrupt_cache_fallback() {
        let fresh_cache = r#"{
          "deepseek": {
            "npm": "@ai-sdk/deepseek",
            "models": {
              "deepseek-v5": {
                "tool_call": true,
                "limit": { "context": 256000, "output": 64000 },
                "cost": { "input": 0.25, "output": 0.50, "cache_read": 0.025 }
              }
            }
          }
        }"#;

        let overrides = [ModelPricing {
            provider_id: "deepseek".into(),
            model: "deepseek-v4-pro".into(),
            family: ProviderFamily::OpenAiChat,
            context_window: 512_000,
            max_output_tokens: 32_768,
            input_per_million_usd: 1.0,
            output_per_million_usd: 2.0,
            cache_read_per_million_usd: 0.1,
            cache_write_per_million_usd: 0.0,
            supports_tool_call: true,
            supports_reasoning: true,
        }];

        let layered = PricingCatalog::layered(Some(fresh_cache), overrides);
        assert_eq!(layered.context_window("deepseek-v5"), Some(256_000));
        assert_eq!(layered.context_window("DEEPSEEK-V4-PRO"), Some(512_000));
        assert_eq!(layered.context_window("claude-sonnet-4-6"), Some(200_000));

        let corrupt_fallback = PricingCatalog::layered(Some("{not valid json"), []);
        assert_eq!(
            corrupt_fallback.context_window("claude-sonnet-4-6"),
            Some(200_000)
        );
    }

    #[test]
    fn estimate_cost_separates_uncached_cache_read_cache_write_and_savings() {
        let catalog = PricingCatalog::builtin();
        let usage = TokenUsage {
            input_tokens: 1_000_000,
            output_tokens: 100_000,
            cache_read_tokens: 700_000,
            cache_write_tokens: 200_000,
            reasoning_tokens: 0,
        };

        // Sonnet 4.6: $3/M uncached, $0.30/M cache_read, $3.75/M cache_write, $15/M output
        // Uncached = 100k -> $0.30; Cache read = 700k -> $0.21;
        // Cache write = 200k -> $0.75; Output = 100k -> $1.50; Total = $2.76; Savings = $1.89
        let cost = catalog
            .estimate_cost("claude-sonnet-4-6", &usage)
            .expect("cost breakdown");
        assert!((cost.uncached_input_usd - 0.30).abs() < 1e-9);
        assert!((cost.cache_read_usd - 0.21).abs() < 1e-9);
        assert!((cost.cache_write_usd - 0.75).abs() < 1e-9);
        assert!((cost.output_usd - 1.50).abs() < 1e-9);
        assert!((cost.total_usd - 2.76).abs() < 1e-9);
        assert!((cost.cache_savings_usd - 1.89).abs() < 1e-9);

        let skewed = TokenUsage {
            input_tokens: 100,
            output_tokens: 0,
            cache_read_tokens: 500,
            cache_write_tokens: 0,
            reasoning_tokens: 0,
        };
        let skewed_cost = catalog
            .estimate_cost("claude-sonnet-4-6", &skewed)
            .expect("skewed usage");
        assert!((skewed_cost.uncached_input_usd - 0.0).abs() < 1e-9);
    }
}
