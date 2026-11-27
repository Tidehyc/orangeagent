use serde::{Deserialize, Serialize};

use crate::{ErrorCategory, ErrorVerdict, ProviderFamily};

/// Functional role used to select a statically pinned model tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    /// Primary coding turn model (pinned per session to preserve prompt cache).
    Main,
    /// Lightweight model for parallel read-only subagent exploration.
    ExploreSubagent,
    /// Fast model dedicated to context summarization / compaction.
    Compactor,
}

/// Concrete provider + model binding for a role or fallback slot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelTarget {
    pub family: ProviderFamily,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget_tokens: Option<u32>,
}

impl ModelTarget {
    pub fn new(family: ProviderFamily, model: impl Into<String>) -> Self {
        Self {
            family,
            model: model.into(),
            max_tokens: None,
            thinking_budget_tokens: None,
        }
    }

    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    #[must_use]
    pub fn with_thinking_budget(mut self, budget_tokens: u32) -> Self {
        self.thinking_budget_tokens = Some(budget_tokens);
        self
    }
}

/// Static role-based model routing profile.
///
/// Design rule: `Main` stays pinned within an active session so prefix caches
/// never thrash; `ExploreSubagent` and `Compactor` route to cheaper/faster tiers;
/// `fallbacks` are only invoked on transient/provider-level failures.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleModelProfile {
    pub main: ModelTarget,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explore_subagent: Option<ModelTarget>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compactor: Option<ModelTarget>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fallbacks: Vec<ModelTarget>,
}

impl RoleModelProfile {
    /// Build a profile where every role uses the same primary [`ModelTarget`].
    pub fn single(main: ModelTarget) -> Self {
        Self {
            main,
            explore_subagent: None,
            compactor: None,
            fallbacks: Vec::new(),
        }
    }

    /// Resolve the configured target for `role`, falling back to `main` when unset.
    pub fn target_for(&self, role: ModelRole) -> &ModelTarget {
        match role {
            ModelRole::Main => &self.main,
            ModelRole::ExploreSubagent => self.explore_subagent.as_ref().unwrap_or(&self.main),
            ModelRole::Compactor => self.compactor.as_ref().unwrap_or(&self.main),
        }
    }

    /// Return ordered fallback candidates excluding `failed_model`.
    pub fn fallback_candidates(&self, failed_model: &str) -> Vec<&ModelTarget> {
        self.fallbacks
            .iter()
            .filter(|candidate| candidate.model != failed_model)
            .collect()
    }

    /// Pick the first eligible fallback target given a failed model name and [`ErrorVerdict`].
    ///
    /// Context overflows, user cancellations, invalid requests, and protocol errors
    /// do not fall back to another model because switching models won't cure them.
    pub fn next_fallback(&self, failed_model: &str, verdict: ErrorVerdict) -> Option<&ModelTarget> {
        let eligible = matches!(
            verdict.category,
            ErrorCategory::RateLimit
                | ErrorCategory::ServerError
                | ErrorCategory::TransientNetwork
                | ErrorCategory::Auth
        );
        if !eligible {
            return None;
        }
        self.fallback_candidates(failed_model).into_iter().next()
    }

    /// Whether falling back from `from` to `to` crosses protocol families
    /// and therefore requires [`crate::sanitize_for_provider`].
    pub fn requires_cross_provider_sanitization(from: ProviderFamily, to: ProviderFamily) -> bool {
        from != to
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_profile_routes_each_role_and_defaults_to_main() {
        let single = RoleModelProfile::single(
            ModelTarget::new(ProviderFamily::Anthropic, "claude-sonnet-4-6")
                .with_max_tokens(16_384)
                .with_thinking_budget(4_096),
        );
        assert_eq!(
            single.target_for(ModelRole::Main).model,
            "claude-sonnet-4-6"
        );
        assert_eq!(
            single.target_for(ModelRole::ExploreSubagent).model,
            "claude-sonnet-4-6"
        );
        assert_eq!(
            single.target_for(ModelRole::Compactor).model,
            "claude-sonnet-4-6"
        );

        let tiered = RoleModelProfile {
            main: ModelTarget::new(ProviderFamily::Anthropic, "claude-opus-4-6"),
            explore_subagent: Some(ModelTarget::new(
                ProviderFamily::Anthropic,
                "claude-haiku-4-5",
            )),
            compactor: Some(ModelTarget::new(
                ProviderFamily::OpenAiChat,
                "deepseek-chat",
            )),
            fallbacks: vec![],
        };
        assert_eq!(tiered.target_for(ModelRole::Main).model, "claude-opus-4-6");
        assert_eq!(
            tiered.target_for(ModelRole::ExploreSubagent).model,
            "claude-haiku-4-5"
        );
        assert_eq!(
            tiered.target_for(ModelRole::Compactor).model,
            "deepseek-chat"
        );
    }

    #[test]
    fn fallback_selection_respects_error_verdict_and_skips_failed_model() {
        let profile = RoleModelProfile {
            main: ModelTarget::new(ProviderFamily::Anthropic, "claude-sonnet-4-6"),
            explore_subagent: None,
            compactor: None,
            fallbacks: vec![
                ModelTarget::new(ProviderFamily::Anthropic, "claude-sonnet-4-6"),
                ModelTarget::new(ProviderFamily::OpenAiChat, "deepseek-v4-pro"),
                ModelTarget::new(ProviderFamily::OpenAiChat, "gpt-4.1"),
            ],
        };

        let rate_limit = ErrorVerdict {
            category: ErrorCategory::RateLimit,
            retryable: true,
            should_compress: false,
            retry_after_ms: Some(1_000),
        };
        let picked = profile
            .next_fallback("claude-sonnet-4-6", rate_limit)
            .expect("should pick deepseek-v4-pro skipping failed main");
        assert_eq!(picked.model, "deepseek-v4-pro");
        assert!(RoleModelProfile::requires_cross_provider_sanitization(
            profile.main.family,
            picked.family
        ));
        assert!(!RoleModelProfile::requires_cross_provider_sanitization(
            ProviderFamily::Anthropic,
            ProviderFamily::Anthropic
        ));

        let overflow = ErrorVerdict {
            category: ErrorCategory::ContextOverflow,
            retryable: true,
            should_compress: true,
            retry_after_ms: None,
        };
        assert!(
            profile
                .next_fallback("claude-sonnet-4-6", overflow)
                .is_none()
        );

        let cancelled = ErrorVerdict {
            category: ErrorCategory::Cancelled,
            retryable: false,
            should_compress: false,
            retry_after_ms: None,
        };
        assert!(
            profile
                .next_fallback("claude-sonnet-4-6", cancelled)
                .is_none()
        );

        let json = serde_json::to_string(&profile).expect("serialize profile");
        let decoded: RoleModelProfile = serde_json::from_str(&json).expect("deserialize profile");
        assert_eq!(decoded, profile);
    }
}
