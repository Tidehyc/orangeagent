use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read as _, Seek as _, SeekFrom, Write as _};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use orangeagent_ai::{Message, TokenUsage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Current JSONL session schema version.
pub const SESSION_SCHEMA_VERSION: u32 = 1;

/// Errors returned by [`JsonlSessionStore`].
#[derive(Debug, Error)]
pub enum SessionStoreError {
    #[error("session I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("session serialization error: {0}")]
    Serde(#[from] serde_json::Error),

    #[error("corrupted JSONL entry at line {line}: {message}")]
    CorruptedEntry { line: usize, message: String },

    #[error("unknown session entry id `{0}`")]
    UnknownEntry(String),

    #[error("parent cycle detected at entry id `{0}`")]
    CycleDetected(String),

    #[error("invalid session operation: {0}")]
    InvalidOperation(String),
}

/// Payload variant stored in a single append-only JSONL [`SessionEntry`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EntryKind {
    /// Root metadata entry written once when a session is created.
    SessionHeader {
        schema_version: u32,
        session_id: String,
        workspace_root: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },

    /// A conversation turn message (`User`, `Assistant`, `Tool`, or `System`).
    Message {
        message: Message,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<TokenUsage>,
        /// Optional Shadow-Git checkpoint commit SHA captured at this point (Phase M2).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        checkpoint_sha: Option<String>,
    },

    /// Context compaction boundary (Phase M2): replaces older history with a structured summary
    /// while preserving the protected tail starting at `first_kept_entry_id`.
    Compaction {
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        first_kept_entry_id: Option<String>,
        tokens_before: u64,
    },

    /// Non-destructive branch switch marker (`/branch` or `/tree`).
    ///
    /// Moves `active_leaf_id` back to `target_leaf_id` so subsequent appends fork from that
    /// ancestor without deleting the previous branch.
    BranchSwitch {
        target_leaf_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        label: Option<String>,
    },

    /// Session metadata update (such as updating the human-readable title).
    Metadata {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },
}

/// One line in an append-only tree-structured session JSONL file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub timestamp_ms: u64,
    pub kind: EntryKind,
}

/// Summary of opening a JSONL session file, including crash-recovery repair telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadReport {
    pub entries_loaded: usize,
    pub truncated_tail_bytes: u64,
    pub repaired: bool,
}

/// Tree-structured, append-only JSONL session store (Source of Truth).
///
/// Every entry carries `(id, parent_id)` so conversations form a DAG/tree supporting
/// non-destructive `/branch` forks and `/tree` time travel. Opening a file automatically
/// detects and truncates a half-written trailing line left behind by a crash or `SIGKILL`,
/// while failing loudly on mid-file corruption.
#[derive(Debug)]
pub struct JsonlSessionStore {
    path: PathBuf,
    entries: HashMap<String, SessionEntry>,
    order: Vec<String>,
    session_id: Option<String>,
    workspace_root: Option<String>,
    title: Option<String>,
    active_leaf_id: Option<String>,
    next_seq: u64,
    load_report: LoadReport,
}

impl JsonlSessionStore {
    /// Create a brand-new JSONL session file and write its root [`EntryKind::SessionHeader`].
    pub fn create(
        path: impl AsRef<Path>,
        session_id: impl Into<String>,
        workspace_root: impl Into<String>,
        title: Option<String>,
    ) -> Result<Self, SessionStoreError> {
        let path = path.as_ref().to_path_buf();
        if path.exists() {
            return Err(SessionStoreError::InvalidOperation(format!(
                "session file `{}` already exists",
                path.display()
            )));
        }

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }

        // Create the empty file first.
        File::create(&path)?;

        let mut store = Self {
            path,
            entries: HashMap::new(),
            order: Vec::new(),
            session_id: None,
            workspace_root: None,
            title: None,
            active_leaf_id: None,
            next_seq: 1,
            load_report: LoadReport::default(),
        };

        let header = EntryKind::SessionHeader {
            schema_version: SESSION_SCHEMA_VERSION,
            session_id: session_id.into(),
            workspace_root: workspace_root.into(),
            title,
        };
        store.append_entry(None, header)?;
        Ok(store)
    }

    /// Open an existing JSONL session file, automatically repairing a crash-truncated last line.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SessionStoreError> {
        let path = path.as_ref().to_path_buf();
        let mut raw = Vec::new();
        File::open(&path)?.read_to_end(&mut raw)?;

        let mut store = Self {
            path: path.clone(),
            entries: HashMap::new(),
            order: Vec::new(),
            session_id: None,
            workspace_root: None,
            title: None,
            active_leaf_id: None,
            next_seq: 1,
            load_report: LoadReport::default(),
        };

        let parsed = parse_jsonl_with_tail_recovery(&raw)?;
        if parsed.repaired {
            truncate_file(&path, parsed.valid_prefix_bytes)?;
        }

        for entry in parsed.entries {
            store.apply_in_memory(entry);
        }

        store.load_report = LoadReport {
            entries_loaded: store.order.len(),
            truncated_tail_bytes: parsed.truncated_tail_bytes,
            repaired: parsed.repaired,
        };

        Ok(store)
    }

    /// Open `path` if it exists, otherwise create a new session file with the given metadata.
    pub fn open_or_create(
        path: impl AsRef<Path>,
        session_id: impl Into<String>,
        workspace_root: impl Into<String>,
        title: Option<String>,
    ) -> Result<Self, SessionStoreError> {
        if path.as_ref().exists() {
            Self::open(path)
        } else {
            Self::create(path, session_id, workspace_root, title)
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }

    pub fn workspace_root(&self) -> Option<&str> {
        self.workspace_root.as_deref()
    }

    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    pub fn active_leaf_id(&self) -> Option<&str> {
        self.active_leaf_id.as_deref()
    }

    pub fn load_report(&self) -> &LoadReport {
        &self.load_report
    }

    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&SessionEntry> {
        self.entries.get(id)
    }

    /// Iterate all entries in exact append order (used for SQLite index rebuilds).
    pub fn entries(&self) -> impl Iterator<Item = &SessionEntry> {
        self.order.iter().filter_map(|id| self.entries.get(id))
    }

    /// Append a conversation [`Message`] as a child of the current `active_leaf_id`.
    pub fn append_message(
        &mut self,
        message: Message,
        model: Option<String>,
        usage: Option<TokenUsage>,
        checkpoint_sha: Option<String>,
    ) -> Result<&SessionEntry, SessionStoreError> {
        let parent_id = self.active_leaf_id.clone();
        self.append_entry(
            parent_id,
            EntryKind::Message {
                message,
                model,
                usage,
                checkpoint_sha,
            },
        )
    }

    /// Append a compaction boundary node as a child of the current `active_leaf_id`.
    pub fn append_compaction(
        &mut self,
        summary: impl Into<String>,
        first_kept_entry_id: Option<String>,
        tokens_before: u64,
    ) -> Result<&SessionEntry, SessionStoreError> {
        if let Some(ref kept_id) = first_kept_entry_id
            && !self.entries.contains_key(kept_id)
        {
            return Err(SessionStoreError::UnknownEntry(kept_id.clone()));
        }

        let parent_id = self.active_leaf_id.clone();
        self.append_entry(
            parent_id,
            EntryKind::Compaction {
                summary: summary.into(),
                first_kept_entry_id,
                tokens_before,
            },
        )
    }

    /// Non-destructively switch `active_leaf_id` to `target_leaf_id` so subsequent messages fork
    /// from that node while preserving all existing branches.
    pub fn switch_branch(
        &mut self,
        target_leaf_id: impl Into<String>,
        label: Option<String>,
    ) -> Result<&SessionEntry, SessionStoreError> {
        let target = target_leaf_id.into();
        let Some(target_entry) = self.entries.get(&target) else {
            return Err(SessionStoreError::UnknownEntry(target));
        };
        if matches!(
            target_entry.kind,
            EntryKind::BranchSwitch { .. } | EntryKind::Metadata { .. }
        ) {
            return Err(SessionStoreError::InvalidOperation(format!(
                "entry `{target}` is a control entry and cannot be a branch target"
            )));
        }

        let parent_id = Some(target.clone());
        self.append_entry(
            parent_id,
            EntryKind::BranchSwitch {
                target_leaf_id: target,
                label,
            },
        )
    }

    /// Append a metadata entry updating the session title.
    pub fn update_title(
        &mut self,
        title: impl Into<String>,
    ) -> Result<&SessionEntry, SessionStoreError> {
        let parent_id = self.active_leaf_id.clone();
        self.append_entry(
            parent_id,
            EntryKind::Metadata {
                title: Some(title.into()),
            },
        )
    }

    /// Reconstruct the root-to-leaf chain of entries ending at `leaf_id` (or `active_leaf_id`).
    pub fn branch_entries(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<&SessionEntry>, SessionStoreError> {
        let Some(start_id) = leaf_id.or(self.active_leaf_id.as_deref()) else {
            return Ok(Vec::new());
        };

        let mut chain = Vec::new();
        let mut visited = HashSet::new();
        let mut cursor = Some(start_id);

        while let Some(id) = cursor {
            if !visited.insert(id) {
                return Err(SessionStoreError::CycleDetected(id.to_owned()));
            }
            let entry = self
                .entries
                .get(id)
                .ok_or_else(|| SessionStoreError::UnknownEntry(id.to_owned()))?;
            chain.push(entry);
            cursor = entry.parent_id.as_deref();
        }

        chain.reverse();
        Ok(chain)
    }

    /// Reconstruct the ordered [`Message`] list for LLM context along `leaf_id` (or `active_leaf_id`).
    ///
    /// If the branch contains [`EntryKind::Compaction`] entries, reconstruction starts at the
    /// most recent compaction summary, followed by any protected tail starting from
    /// `first_kept_entry_id` and all messages appended after the compaction node.
    pub fn branch_messages(
        &self,
        leaf_id: Option<&str>,
    ) -> Result<Vec<Message>, SessionStoreError> {
        let chain = self.branch_entries(leaf_id)?;

        let latest_compaction =
            chain
                .iter()
                .enumerate()
                .rev()
                .find_map(|(idx, entry)| match &entry.kind {
                    EntryKind::Compaction {
                        summary,
                        first_kept_entry_id,
                        ..
                    } => Some((idx, summary.as_str(), first_kept_entry_id.as_deref())),
                    _ => None,
                });

        let mut messages = Vec::new();

        if let Some((compaction_idx, summary, first_kept)) = latest_compaction {
            messages.push(Message::user(format!(
                "<compaction_summary>\n{summary}\n</compaction_summary>"
            )));

            if let Some(kept_id) = first_kept
                && let Some(kept_idx) = chain[..compaction_idx]
                    .iter()
                    .position(|entry| entry.id == kept_id)
            {
                for entry in &chain[kept_idx..compaction_idx] {
                    if let EntryKind::Message { message, .. } = &entry.kind {
                        messages.push(message.clone());
                    }
                }
            }

            for entry in &chain[compaction_idx + 1..] {
                if let EntryKind::Message { message, .. } = &entry.kind {
                    messages.push(message.clone());
                }
            }
            return Ok(messages);
        }

        for entry in chain {
            if let EntryKind::Message { message, .. } = &entry.kind {
                messages.push(message.clone());
            }
        }
        Ok(messages)
    }

    /// Return all direct content children (`Message` / `Compaction`) of `parent_id`.
    pub fn children_of(&self, parent_id: &str) -> Vec<&SessionEntry> {
        self.entries()
            .filter(|entry| {
                is_tree_node(&entry.kind) && entry.parent_id.as_deref() == Some(parent_id)
            })
            .collect()
    }

    /// Return all conversation tree leaf IDs (tree nodes that have no content children).
    pub fn leaf_ids(&self) -> Vec<String> {
        let mut parents_with_children: HashSet<&str> = HashSet::new();
        for entry in self.entries() {
            if is_tree_node(&entry.kind)
                && let Some(ref pid) = entry.parent_id
            {
                parents_with_children.insert(pid.as_str());
            }
        }

        self.entries()
            .filter(|entry| {
                is_tree_node(&entry.kind) && !parents_with_children.contains(entry.id.as_str())
            })
            .map(|entry| entry.id.clone())
            .collect()
    }

    fn append_entry(
        &mut self,
        parent_id: Option<String>,
        kind: EntryKind,
    ) -> Result<&SessionEntry, SessionStoreError> {
        let id = self.allocate_id();
        let entry = SessionEntry {
            id: id.clone(),
            parent_id,
            timestamp_ms: now_ms(),
            kind,
        };

        append_line_durable(&self.path, &entry)?;
        self.apply_in_memory(entry);

        self.entries
            .get(&id)
            .ok_or(SessionStoreError::UnknownEntry(id))
    }

    fn allocate_id(&mut self) -> String {
        loop {
            let candidate = format!("e{:06}", self.next_seq);
            self.next_seq = self.next_seq.saturating_add(1);
            if !self.entries.contains_key(&candidate) {
                return candidate;
            }
        }
    }

    fn apply_in_memory(&mut self, entry: SessionEntry) {
        if let Some(num) = parse_seq_id(&entry.id) {
            self.next_seq = self.next_seq.max(num.saturating_add(1));
        }

        match &entry.kind {
            EntryKind::SessionHeader {
                session_id,
                workspace_root,
                title,
                ..
            } => {
                self.session_id = Some(session_id.clone());
                self.workspace_root = Some(workspace_root.clone());
                if title.is_some() {
                    self.title.clone_from(title);
                }
                self.active_leaf_id = Some(entry.id.clone());
            }
            EntryKind::Message { .. } | EntryKind::Compaction { .. } => {
                self.active_leaf_id = Some(entry.id.clone());
            }
            EntryKind::BranchSwitch { target_leaf_id, .. } => {
                if self.entries.contains_key(target_leaf_id) {
                    self.active_leaf_id = Some(target_leaf_id.clone());
                }
            }
            EntryKind::Metadata { title } => {
                if title.is_some() {
                    self.title.clone_from(title);
                }
            }
        }

        let id = entry.id.clone();
        if !self.entries.contains_key(&id) {
            self.order.push(id.clone());
        }
        self.entries.insert(id, entry);
    }
}

fn is_tree_node(kind: &EntryKind) -> bool {
    matches!(
        kind,
        EntryKind::SessionHeader { .. } | EntryKind::Message { .. } | EntryKind::Compaction { .. }
    )
}

struct ParsedLog {
    entries: Vec<SessionEntry>,
    valid_prefix_bytes: u64,
    truncated_tail_bytes: u64,
    repaired: bool,
}

fn parse_jsonl_with_tail_recovery(raw: &[u8]) -> Result<ParsedLog, SessionStoreError> {
    let mut entries = Vec::new();
    let mut valid_prefix_bytes = 0usize;
    let mut offset = 0usize;
    let mut line_no = 0usize;

    while offset < raw.len() {
        line_no += 1;
        let line_start = offset;
        let rel_nl = raw[offset..].iter().position(|&b| b == b'\n');
        let line_end = match rel_nl {
            Some(pos) => offset + pos + 1,
            None => raw.len(),
        };
        offset = line_end;

        let line_bytes = &raw[line_start..line_end];
        let trimmed_bytes = trim_ascii_whitespace(line_bytes);
        if trimmed_bytes.is_empty() {
            valid_prefix_bytes = line_end;
            continue;
        }

        let parse_result = std::str::from_utf8(trimmed_bytes)
            .map_err(|err| err.to_string())
            .and_then(|s| serde_json::from_str::<SessionEntry>(s).map_err(|err| err.to_string()));

        match parse_result {
            Ok(entry) => {
                entries.push(entry);
                valid_prefix_bytes = line_end;
            }
            Err(message) => {
                let remaining_non_empty = !trim_ascii_whitespace(&raw[line_end..]).is_empty();
                if remaining_non_empty {
                    return Err(SessionStoreError::CorruptedEntry {
                        line: line_no,
                        message,
                    });
                }

                // Trailing corrupted/partial line (crash or SIGKILL mid-write) -> truncate cleanly.
                let truncated = raw.len().saturating_sub(valid_prefix_bytes);
                return Ok(ParsedLog {
                    entries,
                    valid_prefix_bytes: u64::try_from(valid_prefix_bytes).unwrap_or(u64::MAX),
                    truncated_tail_bytes: u64::try_from(truncated).unwrap_or(u64::MAX),
                    repaired: true,
                });
            }
        }
    }

    Ok(ParsedLog {
        entries,
        valid_prefix_bytes: u64::try_from(valid_prefix_bytes).unwrap_or(u64::MAX),
        truncated_tail_bytes: 0,
        repaired: false,
    })
}

fn trim_ascii_whitespace(mut bytes: &[u8]) -> &[u8] {
    while let Some((&first, rest)) = bytes.split_first() {
        if first.is_ascii_whitespace() {
            bytes = rest;
        } else {
            break;
        }
    }
    while let Some((&last, rest)) = bytes.split_last() {
        if last.is_ascii_whitespace() {
            bytes = rest;
        } else {
            break;
        }
    }
    bytes
}

fn truncate_file(path: &Path, len: u64) -> std::io::Result<()> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.set_len(len)?;
    file.sync_data()?;
    Ok(())
}

fn append_line_durable(path: &Path, entry: &SessionEntry) -> Result<(), SessionStoreError> {
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;

    let len = file.metadata()?.len();
    let needs_leading_newline = if len > 0 {
        file.seek(SeekFrom::End(-1))?;
        let mut last = [0u8; 1];
        file.read_exact(&mut last)?;
        last[0] != b'\n'
    } else {
        false
    };

    let mut payload = String::new();
    if needs_leading_newline {
        payload.push('\n');
    }
    payload.push_str(&serde_json::to_string(entry)?);
    payload.push('\n');

    file.write_all(payload.as_bytes())?;
    file.flush()?;
    file.sync_data()?;
    Ok(())
}

fn parse_seq_id(id: &str) -> Option<u64> {
    id.strip_prefix('e')?.parse::<u64>().ok()
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_append_and_reopen_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("sessions/s1.jsonl");

        let mut store =
            JsonlSessionStore::create(&path, "sess-1", "/repo", Some("Initial task".into()))
                .expect("create");
        assert_eq!(store.session_id(), Some("sess-1"));
        assert_eq!(store.title(), Some("Initial task"));

        let usage = TokenUsage {
            input_tokens: 120,
            output_tokens: 45,
            cache_read_tokens: 80,
            cache_write_tokens: 0,
            reasoning_tokens: 15,
        };
        store
            .append_message(Message::user("hello"), None, None, None)
            .expect("append user");
        let last_id = store
            .append_message(
                Message::assistant_text("hi there"),
                Some("gpt-4o".into()),
                Some(usage),
                Some("abc1234".into()),
            )
            .expect("append assistant")
            .id
            .clone();
        store.update_title("Renamed task").expect("title");

        let reopened = JsonlSessionStore::open(&path).expect("reopen");
        assert_eq!(reopened.session_id(), Some("sess-1"));
        assert_eq!(reopened.workspace_root(), Some("/repo"));
        assert_eq!(reopened.title(), Some("Renamed task"));
        assert_eq!(reopened.active_leaf_id(), Some(last_id.as_str()));
        assert!(!reopened.load_report().repaired);

        let msgs = reopened.branch_messages(None).expect("messages");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text_content(), "hello");
        assert_eq!(msgs[1].text_content(), "hi there");
    }

    #[test]
    fn branch_switch_forks_non_destructively_and_reconstructs_both_branches() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("branch.jsonl");

        let mut store =
            JsonlSessionStore::create(&path, "sess-branch", "/repo", None).expect("new");
        let m1 = store
            .append_message(Message::user("root question"), None, None, None)
            .expect("m1")
            .id
            .clone();
        store
            .append_message(Message::assistant_text("approach A"), None, None, None)
            .expect("m2");
        let first_branch_leaf = store
            .append_message(Message::user("A failed"), None, None, None)
            .expect("m3")
            .id
            .clone();

        // Time-travel back to m1 and fork branch B.
        store
            .switch_branch(&m1, Some("try-b".into()))
            .expect("switch");
        assert_eq!(store.active_leaf_id(), Some(m1.as_str()));

        let second_branch_leaf = store
            .append_message(Message::assistant_text("approach B"), None, None, None)
            .expect("m4")
            .id
            .clone();

        // Active branch is B: [root question, approach B].
        let active_msgs = store.branch_messages(None).expect("branch B");
        let active_texts: Vec<String> = active_msgs.iter().map(Message::text_content).collect();
        assert_eq!(active_texts, vec!["root question", "approach B"]);

        // Branch A is still 100% intact on disk and in memory.
        let branch_a_msgs = store
            .branch_messages(Some(&first_branch_leaf))
            .expect("branch A");
        let a_texts: Vec<String> = branch_a_msgs.iter().map(Message::text_content).collect();
        assert_eq!(a_texts, vec!["root question", "approach A", "A failed"]);

        let leaves = store.leaf_ids();
        assert!(leaves.contains(&first_branch_leaf));
        assert!(leaves.contains(&second_branch_leaf));
        assert_eq!(store.children_of(&m1).len(), 2);

        // Reopen verifies BranchSwitch marker persists active leaf across restarts.
        let reopened = JsonlSessionStore::open(&path).expect("reopen");
        assert_eq!(reopened.active_leaf_id(), Some(second_branch_leaf.as_str()));
        let reopened_texts: Vec<String> = reopened
            .branch_messages(None)
            .expect("reopened msgs")
            .iter()
            .map(Message::text_content)
            .collect();
        assert_eq!(reopened_texts, vec!["root question", "approach B"]);
    }

    #[test]
    fn open_repairs_truncated_trailing_line_and_allows_continued_appends() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("crash.jsonl");

        let mut store = JsonlSessionStore::create(&path, "sess-crash", "/repo", None).expect("new");
        store
            .append_message(Message::user("first"), None, None, None)
            .expect("first");
        store
            .append_message(Message::assistant_text("second"), None, None, None)
            .expect("second");

        // Simulate a SIGKILL / power-loss leaving a half-written JSON tail line.
        let mut raw_file = OpenOptions::new().append(true).open(&path).expect("open");
        raw_file
            .write_all(b"{\"id\":\"e000099\",\"parent_id\":\"e000003\",\"kind\":{\"type\":\"mes")
            .expect("corrupt tail");
        drop(raw_file);

        let mut recovered = JsonlSessionStore::open(&path).expect("recover");
        assert!(recovered.load_report().repaired);
        assert!(recovered.load_report().truncated_tail_bytes > 0);
        assert_eq!(recovered.branch_messages(None).expect("msgs").len(), 2);

        // Continued append after self-healing must succeed cleanly.
        recovered
            .append_message(Message::user("third after recovery"), None, None, None)
            .expect("third");

        let clean = JsonlSessionStore::open(&path).expect("reopen clean");
        assert!(!clean.load_report().repaired);
        let texts: Vec<String> = clean
            .branch_messages(None)
            .expect("clean msgs")
            .iter()
            .map(Message::text_content)
            .collect();
        assert_eq!(texts, vec!["first", "second", "third after recovery"]);
    }

    #[test]
    fn open_rejects_corrupted_middle_line() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad_mid.jsonl");

        let valid_header = SessionEntry {
            id: "e000001".into(),
            parent_id: None,
            timestamp_ms: 1,
            kind: EntryKind::SessionHeader {
                schema_version: SESSION_SCHEMA_VERSION,
                session_id: "s".into(),
                workspace_root: "/repo".into(),
                title: None,
            },
        };
        let valid_msg = SessionEntry {
            id: "e000002".into(),
            parent_id: Some("e000001".into()),
            timestamp_ms: 2,
            kind: EntryKind::Message {
                message: Message::user("ok"),
                model: None,
                usage: None,
                checkpoint_sha: None,
            },
        };

        let content = format!(
            "{}\n{{not valid json\n{}\n",
            serde_json::to_string(&valid_header).expect("ser1"),
            serde_json::to_string(&valid_msg).expect("ser2"),
        );
        std::fs::write(&path, content).expect("write");

        let err = JsonlSessionStore::open(&path).expect_err("must fail on mid-file corruption");
        assert!(matches!(
            err,
            SessionStoreError::CorruptedEntry { line: 2, .. }
        ));
    }

    #[test]
    fn branch_messages_respects_compaction_boundary_and_protected_tail() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("compact.jsonl");

        let mut store = JsonlSessionStore::create(&path, "sess-c", "/repo", None).expect("new");
        store
            .append_message(Message::user("old 1"), None, None, None)
            .expect("m1");
        store
            .append_message(Message::assistant_text("old 2"), None, None, None)
            .expect("m2");
        let kept_id = store
            .append_message(Message::user("protected tail"), None, None, None)
            .expect("m3")
            .id
            .clone();

        store
            .append_compaction("Earlier we explored the codebase.", Some(kept_id), 12_000)
            .expect("compact");
        store
            .append_message(
                Message::assistant_text("continuing after compact"),
                None,
                None,
                None,
            )
            .expect("m4");

        let msgs = store.branch_messages(None).expect("msgs");
        let texts: Vec<String> = msgs.iter().map(Message::text_content).collect();
        assert_eq!(texts.len(), 3);
        assert!(texts[0].contains("Earlier we explored the codebase."));
        assert_eq!(texts[1], "protected tail");
        assert_eq!(texts[2], "continuing after compact");
    }

    #[test]
    fn branch_entries_detects_parent_cycle() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("cycle.jsonl");

        let a = SessionEntry {
            id: "a".into(),
            parent_id: Some("b".into()),
            timestamp_ms: 1,
            kind: EntryKind::Message {
                message: Message::user("a"),
                model: None,
                usage: None,
                checkpoint_sha: None,
            },
        };
        let b = SessionEntry {
            id: "b".into(),
            parent_id: Some("a".into()),
            timestamp_ms: 2,
            kind: EntryKind::Message {
                message: Message::user("b"),
                model: None,
                usage: None,
                checkpoint_sha: None,
            },
        };
        let body = format!(
            "{}\n{}\n",
            serde_json::to_string(&a).expect("a"),
            serde_json::to_string(&b).expect("b")
        );
        std::fs::write(&path, body).expect("write");

        let store = JsonlSessionStore::open(&path).expect("open");
        let err = store
            .branch_entries(Some("b"))
            .expect_err("must detect cycle");
        assert!(matches!(err, SessionStoreError::CycleDetected(_)));
    }
}
