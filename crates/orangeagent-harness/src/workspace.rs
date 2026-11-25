use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Shared tracker of files read during the current session.
///
/// Records a deterministic 64-bit FNV-1a content digest for every file read via `read`,
/// enabling downstream `edit`/`write` tools (Step 7) to block blind edits and detect
/// external modifications between read and write.
#[derive(Debug, Default, Clone)]
pub struct ReadFileTracker {
    hashes: Arc<Mutex<HashMap<PathBuf, u64>>>,
}

impl ReadFileTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that `canonical_path` was read with `content`.
    pub fn record_read(&self, canonical_path: PathBuf, content: &[u8]) {
        let digest = fnv1a_64(content);
        let mut guard = self
            .hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.insert(canonical_path, digest);
    }

    /// Return `true` if `canonical_path` has been read at least once in this session.
    pub fn has_read(&self, canonical_path: &Path) -> bool {
        let guard = self
            .hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.contains_key(canonical_path)
    }

    /// Return the last recorded content hash for `canonical_path`, if any.
    pub fn recorded_hash(&self, canonical_path: &Path) -> Option<u64> {
        let guard = self
            .hashes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        guard.get(canonical_path).copied()
    }

    /// Return `true` iff `canonical_path` was previously read and its current bytes match.
    pub fn matches_current(&self, canonical_path: &Path, current_content: &[u8]) -> bool {
        self.recorded_hash(canonical_path)
            .is_some_and(|recorded| recorded == fnv1a_64(current_content))
    }
}

/// Shared workspace root and read-state handle injected into all harness tools.
#[derive(Debug, Clone)]
pub struct WorkspaceContext {
    root: PathBuf,
    read_tracker: ReadFileTracker,
}

impl WorkspaceContext {
    /// Create a new [`WorkspaceContext`] anchored at `root` (canonicalized on disk).
    pub fn new(root: impl AsRef<Path>) -> std::io::Result<Self> {
        let canonical_root = std::fs::canonicalize(root)?;
        Ok(Self {
            root: canonical_root,
            read_tracker: ReadFileTracker::new(),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn read_tracker(&self) -> &ReadFileTracker {
        &self.read_tracker
    }

    /// Resolve an existing file or directory path, strictly enforcing workspace containment.
    ///
    /// Rejects symlink escapes and `..` traversal outside [`WorkspaceContext::root`].
    pub fn resolve_existing_path(&self, raw: &str) -> Result<PathBuf, String> {
        let candidate = self.join_raw(raw);
        let canonical = std::fs::canonicalize(&candidate)
            .map_err(|err| format!("failed to access path `{raw}`: {err}"))?;
        if !canonical.starts_with(&self.root) {
            return Err(format!(
                "path `{raw}` escapes workspace root `{}`",
                self.root.display()
            ));
        }
        Ok(canonical)
    }

    /// Resolve a target path that may not exist yet (for Step 7 `write`), ensuring the
    /// nearest existing ancestor and lexical target stay strictly inside [`WorkspaceContext::root`].
    pub fn resolve_target_path(&self, raw: &str) -> Result<PathBuf, String> {
        let candidate = self.join_raw(raw);
        let mut existing = candidate.as_path();
        let mut suffix: Vec<std::ffi::OsString> = Vec::new();

        while !existing.exists() {
            let name = existing
                .file_name()
                .ok_or_else(|| format!("invalid target path `{raw}`"))?;
            suffix.push(name.to_os_string());
            existing = existing
                .parent()
                .ok_or_else(|| format!("invalid target path `{raw}`"))?;
        }

        let mut resolved = std::fs::canonicalize(existing)
            .map_err(|err| format!("failed to resolve ancestor of `{raw}`: {err}"))?;
        for part in suffix.into_iter().rev() {
            resolved.push(part);
        }

        let normalized = lexical_normalize(&resolved);
        if !normalized.starts_with(&self.root) {
            return Err(format!(
                "path `{raw}` escapes workspace root `{}`",
                self.root.display()
            ));
        }
        Ok(normalized)
    }

    /// Format a canonical path relative to [`WorkspaceContext::root`] with forward slashes.
    pub fn display_relative(&self, path: &Path) -> String {
        let rel = path.strip_prefix(&self.root).unwrap_or(path);
        rel.to_string_lossy().replace('\\', "/")
    }

    fn join_raw(&self, raw: &str) -> PathBuf {
        let p = Path::new(raw);
        if p.is_absolute() {
            p.to_path_buf()
        } else {
            self.root.join(p)
        }
    }
}

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Deterministic 64-bit FNV-1a hash (stable across Rust versions and runs).
pub(crate) fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_context_blocks_traversal_and_tracks_read_hashes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let file = dir.path().join("hello.txt");
        std::fs::write(&file, b"hello world").expect("write");

        let ctx = WorkspaceContext::new(dir.path()).expect("ctx");
        let resolved = ctx.resolve_existing_path("hello.txt").expect("resolve");
        assert_eq!(ctx.display_relative(&resolved), "hello.txt");

        // Escape attempt must be rejected.
        assert!(ctx.resolve_existing_path("../").is_err());
        assert!(ctx.resolve_target_path("../../outside.txt").is_err());

        // ReadFileTracker detects match vs external modification.
        assert!(!ctx.read_tracker().has_read(&resolved));
        ctx.read_tracker()
            .record_read(resolved.clone(), b"hello world");
        assert!(ctx.read_tracker().has_read(&resolved));
        assert!(
            ctx.read_tracker()
                .matches_current(&resolved, b"hello world")
        );
        assert!(!ctx.read_tracker().matches_current(&resolved, b"modified"));
    }
}
