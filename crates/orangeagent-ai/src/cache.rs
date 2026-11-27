use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::ProviderFamily;

/// Anthropic Messages API allows at most 4 explicit `cache_control` breakpoints per request.
pub const MAX_ANTHROPIC_CACHE_BREAKPOINTS: usize = 4;

/// Prompt cache retention TTL supported by Anthropic's explicit `cache_control`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTtl {
    /// Standard 5-minute ephemeral cache (default, no extra beta header required).
    #[default]
    Ephemeral5m,
    /// Extended 1-hour ephemeral cache (requires `extended-cache-ttl-2025-04-11` beta header).
    Ephemeral1h,
}

impl CacheTtl {
    /// Serialize into Anthropic's wire `cache_control` object.
    pub fn to_anthropic_json(self) -> Value {
        match self {
            Self::Ephemeral5m => json!({ "type": "ephemeral" }),
            Self::Ephemeral1h => json!({ "type": "ephemeral", "ttl": "1h" }),
        }
    }

    /// Whether this TTL requires Anthropic's extended-cache-ttl beta header.
    pub fn requires_extended_ttl_beta(self) -> bool {
        matches!(self, Self::Ephemeral1h)
    }
}

/// Canonical four-breakpoint placement slots for Anthropic prompt caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBreakpointSlot {
    /// End of the static `tools` schema array.
    ToolsEnd,
    /// End of the static `system` prompt blocks.
    SystemEnd,
    /// Penultimate cacheable history turn (anchors prior prefix while tail grows).
    HistoryPenultimate,
    /// Latest cacheable history turn (maximizes hit on next turn's incremental append).
    HistoryLatest,
}

/// Concrete cache breakpoint placement plan for a single request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CachePlan {
    pub ttl: CacheTtl,
    pub mark_last_tool: bool,
    pub mark_system: bool,
    /// Up to 2 ascending indices into the serialized `wire_messages` array.
    pub message_indices: Vec<usize>,
}

impl CachePlan {
    /// Produce a no-op cache plan with 0 breakpoints.
    pub fn disabled() -> Self {
        Self {
            ttl: CacheTtl::default(),
            mark_last_tool: false,
            mark_system: false,
            message_indices: Vec::new(),
        }
    }

    /// Build Anthropic's canonical up-to-4-breakpoint rolling cache plan.
    ///
    /// Layout:
    /// 1. Last tool definition (`ToolsEnd`, if `has_tools`)
    /// 2. Last system block (`SystemEnd`, if `has_system`)
    /// 3. Last up to 2 cacheable message turns in ascending order (`HistoryPenultimate`, `HistoryLatest`)
    pub fn for_anthropic(
        has_tools: bool,
        has_system: bool,
        cacheable_message_indices: &[usize],
        ttl: CacheTtl,
    ) -> Self {
        let len = cacheable_message_indices.len();
        let start = len.saturating_sub(2);
        let message_indices = cacheable_message_indices[start..].to_vec();

        let plan = Self {
            ttl,
            mark_last_tool: has_tools,
            mark_system: has_system,
            message_indices,
        };
        debug_assert!(plan.total_breakpoints() <= MAX_ANTHROPIC_CACHE_BREAKPOINTS);
        plan
    }

    /// Total number of `cache_control` markers this plan will emit.
    pub fn total_breakpoints(&self) -> usize {
        usize::from(self.mark_last_tool)
            + usize::from(self.mark_system)
            + self.message_indices.len()
    }

    /// Return the ordered list of active breakpoint slots in this plan.
    pub fn slots(&self) -> Vec<CacheBreakpointSlot> {
        let mut out = Vec::with_capacity(self.total_breakpoints());
        if self.mark_last_tool {
            out.push(CacheBreakpointSlot::ToolsEnd);
        }
        if self.mark_system {
            out.push(CacheBreakpointSlot::SystemEnd);
        }
        match self.message_indices.len() {
            0 => {}
            1 => out.push(CacheBreakpointSlot::HistoryLatest),
            _ => {
                out.push(CacheBreakpointSlot::HistoryPenultimate);
                out.push(CacheBreakpointSlot::HistoryLatest);
            }
        }
        out
    }
}

/// Whether the given [`ProviderFamily`] uses explicit `cache_control` markers in requests.
pub fn supports_explicit_cache_control(family: ProviderFamily) -> bool {
    matches!(family, ProviderFamily::Anthropic)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_places_up_to_four_breakpoints_in_canonical_order() {
        let p0 = CachePlan::for_anthropic(true, true, &[], CacheTtl::Ephemeral5m);
        assert_eq!(p0.total_breakpoints(), 2);
        assert_eq!(p0.message_indices, Vec::<usize>::new());
        assert_eq!(
            p0.slots(),
            vec![
                CacheBreakpointSlot::ToolsEnd,
                CacheBreakpointSlot::SystemEnd
            ]
        );

        let p1 = CachePlan::for_anthropic(true, true, &[2], CacheTtl::Ephemeral5m);
        assert_eq!(p1.total_breakpoints(), 3);
        assert_eq!(p1.message_indices, vec![2]);
        assert_eq!(
            p1.slots(),
            vec![
                CacheBreakpointSlot::ToolsEnd,
                CacheBreakpointSlot::SystemEnd,
                CacheBreakpointSlot::HistoryLatest,
            ]
        );

        let p2 = CachePlan::for_anthropic(true, true, &[0, 3], CacheTtl::Ephemeral5m);
        assert_eq!(p2.total_breakpoints(), 4);
        assert_eq!(p2.message_indices, vec![0, 3]);

        let p5 = CachePlan::for_anthropic(true, true, &[0, 1, 2, 4, 7], CacheTtl::Ephemeral1h);
        assert_eq!(p5.total_breakpoints(), MAX_ANTHROPIC_CACHE_BREAKPOINTS);
        assert_eq!(p5.message_indices, vec![4, 7]);
        assert_eq!(
            p5.slots(),
            vec![
                CacheBreakpointSlot::ToolsEnd,
                CacheBreakpointSlot::SystemEnd,
                CacheBreakpointSlot::HistoryPenultimate,
                CacheBreakpointSlot::HistoryLatest,
            ]
        );

        let no_tools = CachePlan::for_anthropic(false, false, &[1, 2, 3], CacheTtl::Ephemeral5m);
        assert_eq!(no_tools.total_breakpoints(), 2);
        assert_eq!(no_tools.message_indices, vec![2, 3]);
    }

    #[test]
    fn disabled_plan_and_ttl_serialization() {
        let disabled = CachePlan::disabled();
        assert_eq!(disabled.total_breakpoints(), 0);
        assert_eq!(disabled.slots(), Vec::<CacheBreakpointSlot>::new());

        assert_eq!(
            CacheTtl::Ephemeral5m.to_anthropic_json(),
            json!({ "type": "ephemeral" })
        );
        assert!(!CacheTtl::Ephemeral5m.requires_extended_ttl_beta());

        assert_eq!(
            CacheTtl::Ephemeral1h.to_anthropic_json(),
            json!({ "type": "ephemeral", "ttl": "1h" })
        );
        assert!(CacheTtl::Ephemeral1h.requires_extended_ttl_beta());

        assert!(supports_explicit_cache_control(ProviderFamily::Anthropic));
        assert!(!supports_explicit_cache_control(ProviderFamily::OpenAiChat));
        assert!(!supports_explicit_cache_control(
            ProviderFamily::OpenAiResponses
        ));
        assert!(!supports_explicit_cache_control(ProviderFamily::Gemini));
    }
}
