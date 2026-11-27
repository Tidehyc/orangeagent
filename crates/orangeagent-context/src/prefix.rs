use orangeagent_ai::{ContentBlock, Message, Role, ToolDefinition};
use serde::{Deserialize, Serialize};

/// Lightweight skill descriptor injected into the static system prefix.
///
/// Full `SKILL.md` bodies are intentionally omitted and read on-demand via `read`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SkillSummary {
    pub name: String,
    pub description: String,
}

/// Canonical static system prefix assembled once at session start.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticPrefixSpec {
    pub core_instructions: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agents_md: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skills: Vec<SkillSummary>,
}

impl StaticPrefixSpec {
    /// Render a deterministic system prompt whose byte layout stays identical
    /// across turns (skills sorted alphabetically by name).
    pub fn render_system_prompt(&self) -> String {
        let mut sections = Vec::new();
        let core = self.core_instructions.trim();
        if !core.is_empty() {
            sections.push(core.to_string());
        }

        if let Some(agents) = &self.agents_md {
            let trimmed = agents.trim();
            if !trimmed.is_empty() {
                sections.push(format!(
                    "<project_instructions>\n{trimmed}\n</project_instructions>"
                ));
            }
        }

        if !self.skills.is_empty() {
            let mut sorted = self.skills.clone();
            sorted.sort_by(|a, b| a.name.cmp(&b.name));
            let list: Vec<String> = sorted
                .into_iter()
                .map(|s| format!("- {}: {}", s.name.trim(), s.description.trim()))
                .collect();
            sections.push(format!(
                "<available_skills>\n{}\n</available_skills>",
                list.join("\n")
            ));
        }

        sections.join("\n\n")
    }
}

/// Deterministic FNV-1a 64-bit hashes of the static prefix (`system` + `tools`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefixFingerprint {
    pub system_hash: String,
    pub tools_hash: String,
    pub combined_hash: String,
}

impl PrefixFingerprint {
    pub fn compute(system: Option<&str>, tools: &[ToolDefinition]) -> Self {
        let sys_norm = system.unwrap_or("");
        let system_hash = fnv1a_64_hex(sys_norm.as_bytes());
        let tools_json = serde_json::to_string(tools).unwrap_or_default();
        let tools_hash = fnv1a_64_hex(tools_json.as_bytes());
        let combined_hash = fnv1a_64_hex(format!("{system_hash}:{tools_hash}").as_bytes());
        Self {
            system_hash,
            tools_hash,
            combined_hash,
        }
    }
}

/// Outcome of verifying `(system, tools)` against the session's pinned static prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrefixCheck {
    /// First turn of the session: fingerprint recorded as the baseline.
    Pinned(PrefixFingerprint),
    /// Subsequent turn matches the pinned fingerprint byte-for-byte.
    Stable(PrefixFingerprint),
    /// Prefix changed mid-session — Prompt Cache will be busted from the changed boundary!
    Drifted {
        expected: PrefixFingerprint,
        actual: PrefixFingerprint,
        system_changed: bool,
        tools_changed: bool,
    },
}

impl PrefixCheck {
    pub fn is_stable(&self) -> bool {
        matches!(self, Self::Pinned(_) | Self::Stable(_))
    }
}

/// Session-scoped static prefix invariant guard.
#[derive(Debug, Default, Clone)]
pub struct PrefixGuard {
    pinned: Option<PrefixFingerprint>,
}

impl PrefixGuard {
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin the prefix on the first turn, or verify stability on subsequent turns.
    pub fn verify(&mut self, system: Option<&str>, tools: &[ToolDefinition]) -> PrefixCheck {
        let actual = PrefixFingerprint::compute(system, tools);
        let Some(expected) = &self.pinned else {
            self.pinned = Some(actual.clone());
            return PrefixCheck::Pinned(actual);
        };

        if *expected == actual {
            return PrefixCheck::Stable(actual);
        }

        let system_changed = expected.system_hash != actual.system_hash;
        let tools_changed = expected.tools_hash != actual.tools_hash;
        PrefixCheck::Drifted {
            expected: expected.clone(),
            actual,
            system_changed,
            tools_changed,
        }
    }

    /// Clear the pinned fingerprint (e.g., on explicit session reset).
    pub fn reset(&mut self) {
        self.pinned = None;
    }
}

/// Heuristic lint detecting common cache-busting dynamic data inside `system`.
pub fn detect_volatile_system_prompt(system: &str) -> Vec<&'static str> {
    let mut warnings = Vec::new();
    let lower = system.to_ascii_lowercase();

    if contains_iso_timestamp(system) {
        warnings.push("iso_timestamp");
    }
    if lower.contains("current time:") || lower.contains("today's date:") {
        warnings.push("clock_marker");
    }
    if lower.contains("git status") || lower.contains("git diff") {
        warnings.push("dynamic_git_state");
    }
    warnings
}

/// Append dynamic runtime state (such as `<recovery_drift>`) to the tail of a
/// `Role::User` message so the static `system` prefix remains completely untouched.
pub fn attach_dynamic_reminder(user_message: &mut Message, tag: &str, body: &str) {
    let trimmed = body.trim();
    if trimmed.is_empty() {
        return;
    }
    let block_text = format!("<{tag}>\n{trimmed}\n</{tag}>");
    if user_message.role == Role::User
        && let Some(ContentBlock::Text { text }) = user_message.content.last_mut()
    {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str(&block_text);
        return;
    }
    user_message
        .content
        .push(ContentBlock::Text { text: block_text });
}

fn contains_iso_timestamp(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.len() < 16 {
        return false;
    }
    bytes.windows(16).any(|w| {
        w[0] == b'2'
            && w[1] == b'0'
            && w[2].is_ascii_digit()
            && w[3].is_ascii_digit()
            && w[4] == b'-'
            && w[5].is_ascii_digit()
            && w[6].is_ascii_digit()
            && w[7] == b'-'
            && w[8].is_ascii_digit()
            && w[9].is_ascii_digit()
            && w[10] == b'T'
            && w[11].is_ascii_digit()
            && w[12].is_ascii_digit()
            && w[13] == b':'
    })
}

fn fnv1a_64_hex(bytes: &[u8]) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn prefix_guard_detects_system_and_tool_drift_and_dynamic_tail_preserves_prefix() {
        let spec = StaticPrefixSpec {
            core_instructions: "You are OrangeAgent.".into(),
            agents_md: Some("Always run cargo test.".into()),
            skills: vec![
                SkillSummary {
                    name: "zeta".into(),
                    description: "Zeta skill".into(),
                },
                SkillSummary {
                    name: "alpha".into(),
                    description: "Alpha skill".into(),
                },
            ],
        };
        let system = spec.render_system_prompt();
        let alpha_pos = system.find("alpha:").expect("alpha");
        let zeta_pos = system.find("zeta:").expect("zeta");
        assert!(alpha_pos < zeta_pos);

        let tools = vec![ToolDefinition {
            name: "read".into(),
            description: "Read file".into(),
            parameters: json!({ "type": "object" }),
        }];

        let mut guard = PrefixGuard::new();
        assert!(matches!(
            guard.verify(Some(&system), &tools),
            PrefixCheck::Pinned(_)
        ));
        assert!(matches!(
            guard.verify(Some(&system), &tools),
            PrefixCheck::Stable(_)
        ));

        let mut user_msg = Message::user("continue fixing bug");
        attach_dynamic_reminder(&mut user_msg, "recovery_drift", "M src/lib.rs");
        assert!(user_msg.text_content().contains("<recovery_drift>"));
        // System prefix stays unchanged -> guard stays Stable
        assert!(guard.verify(Some(&system), &tools).is_stable());

        let drifted_sys = format!("{system}\nCurrent time: 2026-11-27T20:00:00Z");
        assert_eq!(
            detect_volatile_system_prompt(&drifted_sys),
            vec!["iso_timestamp", "clock_marker"]
        );
        assert!(matches!(
            guard.verify(Some(&drifted_sys), &tools),
            PrefixCheck::Drifted {
                system_changed: true,
                tools_changed: false,
                ..
            }
        ));
    }
}
