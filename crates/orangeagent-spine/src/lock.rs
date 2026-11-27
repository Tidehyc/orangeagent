use std::fs::{self, File, OpenOptions, TryLockError};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use crate::SpineError;

/// Metadata written inside a locked lease file for observability and diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockLease {
    pub owner_pid: u32,
    pub task_name: String,
    pub acquired_at_ms: u64,
}

/// Cross-process non-blocking exclusive advisory lock backed by OS `flock`
/// (`std::fs::File::try_lock`).
///
/// The kernel automatically releases the lock if the owning process exits or crashes,
/// eliminating stale lockfiles when scheduling shared Cron / Daemon tasks.
#[derive(Debug)]
pub struct ProcessFileLock {
    file: File,
    path: PathBuf,
    lease: LockLease,
}

impl ProcessFileLock {
    /// Attempt a non-blocking exclusive lock on `path` and persist lease metadata.
    ///
    /// Opens the file **without** truncating first so contenders returning
    /// [`SpineError::LockContended`] never clobber the active holder's lease.
    pub fn try_acquire(
        path: impl AsRef<Path>,
        task_name: impl Into<String>,
    ) -> Result<Self, SpineError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|source| SpineError::LockIo {
                path: path.clone(),
                source,
            })?;
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(|source| SpineError::LockIo {
                path: path.clone(),
                source,
            })?;

        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) => return Err(SpineError::LockContended { path }),
            Err(TryLockError::Error(source)) => return Err(SpineError::LockIo { path, source }),
        }

        let lease = LockLease {
            owner_pid: std::process::id(),
            task_name: task_name.into(),
            acquired_at_ms: now_ms(),
        };

        let encoded = serde_json::to_vec(&lease).map_err(|err| SpineError::LockIo {
            path: path.clone(),
            source: std::io::Error::other(err),
        })?;

        file.set_len(0).map_err(|source| SpineError::LockIo {
            path: path.clone(),
            source,
        })?;
        file.seek(SeekFrom::Start(0))
            .map_err(|source| SpineError::LockIo {
                path: path.clone(),
                source,
            })?;
        file.write_all(&encoded)
            .map_err(|source| SpineError::LockIo {
                path: path.clone(),
                source,
            })?;
        file.sync_data().map_err(|source| SpineError::LockIo {
            path: path.clone(),
            source,
        })?;

        Ok(Self { file, path, lease })
    }

    /// Inspect the lease JSON stored in `path` without taking ownership of the lock.
    pub fn read_lease(path: impl AsRef<Path>) -> Result<Option<LockLease>, SpineError> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(None);
        }
        let mut raw = String::new();
        File::open(path)
            .and_then(|mut f| f.read_to_string(&mut raw))
            .map_err(|source| SpineError::LockIo {
                path: path.to_path_buf(),
                source,
            })?;
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Ok(None);
        }
        let lease = serde_json::from_str(trimmed).map_err(|err| SpineError::LockIo {
            path: path.to_path_buf(),
            source: std::io::Error::new(std::io::ErrorKind::InvalidData, err),
        })?;
        Ok(Some(lease))
    }

    pub fn lease(&self) -> &LockLease {
        &self.lease
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ProcessFileLock {
    fn drop(&mut self) {
        let _ = self.file.unlock();
    }
}

fn now_ms() -> u64 {
    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    u64::try_from(dur.as_millis()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use tempfile::TempDir;

    use super::*;

    const PROBE_PATH_ENV: &str = "ORANGEAGENT_LOCK_PROBE_PATH";
    const PROBE_EXPECT_ENV: &str = "ORANGEAGENT_LOCK_PROBE_EXPECT";

    #[test]
    fn process_file_lock_blocks_contenders_and_persists_lease_metadata() {
        let dir = TempDir::new().expect("tempdir");
        let lock_path = dir.path().join("locks").join("cron-nightly.lock");

        let holder = ProcessFileLock::try_acquire(&lock_path, "nightly_cleanup")
            .expect("initial acquire succeeds");
        assert_eq!(holder.lease().owner_pid, std::process::id());
        assert_eq!(holder.lease().task_name, "nightly_cleanup");
        assert_eq!(holder.path(), lock_path);

        let lease_on_disk = ProcessFileLock::read_lease(&lock_path)
            .expect("read lease")
            .expect("lease exists");
        assert_eq!(lease_on_disk.owner_pid, std::process::id());

        let contender = ProcessFileLock::try_acquire(&lock_path, "duplicate_runner");
        assert!(matches!(contender, Err(SpineError::LockContended { .. })));

        // Contender failure must never overwrite or truncate the active holder's lease
        let still_holder = ProcessFileLock::read_lease(&lock_path)
            .expect("read lease after contention")
            .expect("lease still intact");
        assert_eq!(still_holder.task_name, "nightly_cleanup");

        drop(holder);
        let next = ProcessFileLock::try_acquire(&lock_path, "next_runner")
            .expect("lock available after drop");
        assert_eq!(next.lease().task_name, "next_runner");
    }

    #[test]
    fn multi_process_flock_prevents_concurrent_claim_and_releases_on_exit() {
        let dir = TempDir::new().expect("tempdir");
        let lock_path = dir.path().join("daemon-shared.lock");
        let exe = std::env::current_exe().expect("current test binary");

        let holder =
            ProcessFileLock::try_acquire(&lock_path, "parent_daemon").expect("parent acquires");

        let contended_status = Command::new(&exe)
            .args([
                "--exact",
                "lock::tests::child_lock_probe_helper",
                "--nocapture",
            ])
            .env(PROBE_PATH_ENV, &lock_path)
            .env(PROBE_EXPECT_ENV, "contended")
            .status()
            .expect("spawn contended child probe");
        assert!(
            contended_status.success(),
            "child process should observe LockContended while parent holds flock"
        );

        drop(holder);

        let acquired_status = Command::new(&exe)
            .args([
                "--exact",
                "lock::tests::child_lock_probe_helper",
                "--nocapture",
            ])
            .env(PROBE_PATH_ENV, &lock_path)
            .env(PROBE_EXPECT_ENV, "acquire")
            .status()
            .expect("spawn post-drop child probe");
        assert!(
            acquired_status.success(),
            "child process should acquire flock after parent drops it"
        );

        // Once the child process exits, the kernel releases its lock so parent can re-acquire
        let reacquired = ProcessFileLock::try_acquire(&lock_path, "parent_again")
            .expect("parent re-acquires after child exits");
        assert_eq!(reacquired.lease().task_name, "parent_again");
    }

    #[test]
    fn child_lock_probe_helper() {
        let Ok(path) = std::env::var(PROBE_PATH_ENV) else {
            return;
        };
        let expect = std::env::var(PROBE_EXPECT_ENV).unwrap_or_default();
        match expect.as_str() {
            "contended" => {
                let res = ProcessFileLock::try_acquire(&path, "child_probe");
                assert!(
                    matches!(res, Err(SpineError::LockContended { .. })),
                    "expected LockContended, got {res:?}"
                );
            }
            "acquire" => {
                let lock = ProcessFileLock::try_acquire(&path, "child_probe")
                    .expect("child probe acquires released lock");
                assert_eq!(lock.lease().owner_pid, std::process::id());
            }
            other => panic!("unexpected {PROBE_EXPECT_ENV}: {other}"),
        }
    }
}
