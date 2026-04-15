#!/usr/bin/env python3
"""
vault-watcher — Real-time file watcher for vault-memory embeddings.

Monitors directories for .md file changes and triggers immediate
re-indexing + embedding via vault-memory. Designed to run as a
PM2-managed daemon alongside the nightly cron safety net.

Key features:
  - Debouncing: coalesces rapid changes (e.g. bulk copy) into single index ops
  - Locking: prevents concurrent indexing of the same file
  - Graceful: SQLite WAL mode allows concurrent reads during writes
  - Resilient: logs errors, never crashes on single-file failures

Usage:
  python3 vault_watcher.py                    # foreground
  pm2 start vault_watcher.py --name vault-watcher --interpreter python3

Environment:
  NVIDIA_API_KEY       — required for embeddings
  VAULT_MEMORY_DB      — path to vectors.db (default: ~/.hermes/data/vectors.db)
  VAULT_WATCHER_DEBOUNCE — debounce seconds (default: 5)
  VAULT_WATCHER_LOG    — log file path (default: stderr)
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

# Add vault-memory to path
sys.path.insert(0, str(Path(__file__).parent))
from vault_memory import VaultMemory

# ── Configuration ──────────────────────────────────────────────────────────────

WATCH_DIRS: list[dict] = [
    {
        "path": os.path.expanduser("~/obsidian-vault"),
        "collection": "obsidian",
        "glob": "**/*.md",
    },
    {
        "path": os.path.expanduser("~/.hermes/skills"),
        "collection": "hermes-memory",
        "glob": "**/*.md",
    },
    {
        "path": os.path.expanduser("~/.hermes/memories"),
        "collection": "hermes-memory",
        "glob": "**/*.md",
    },
]

DEFAULT_DB = os.path.expanduser(
    os.environ.get("VAULT_MEMORY_DB", "~/.hermes/data/vectors.db")
)
NVIDIA_KEY = os.environ.get("NVIDIA_API_KEY", "")
DEBOUNCE_SECONDS = float(os.environ.get("VAULT_WATCHER_DEBOUNCE", "5"))
# Maximum batch size before forcing a flush (prevents unbounded accumulation)
MAX_BATCH_SIZE = 50
# How often to check for pending work (seconds)
FLUSH_CHECK_INTERVAL = 1.0

# ── Logging ────────────────────────────────────────────────────────────────────

log_file = os.environ.get("VAULT_WATCHER_LOG", "")
log_handlers: list[logging.Handler] = []
if log_file:
    log_handlers.append(logging.FileHandler(log_file))
else:
    log_handlers.append(logging.StreamHandler(sys.stderr))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=log_handlers,
)
logger = logging.getLogger("vault-watcher")


# ── Debounced Indexer ──────────────────────────────────────────────────────────

class DebouncedIndexer:
    """Collects file change events and processes them in batches after a
    debounce window. Thread-safe.

    When files change rapidly (bulk copy, git pull, Syncthing sync),
    events are coalesced: only the last event per file matters, and
    indexing happens once after the storm settles.
    """

    def __init__(self, db_path: str, nvidia_key: str, debounce_s: float):
        self._db_path = db_path
        self._nvidia_key = nvidia_key
        self._debounce_s = debounce_s

        # Pending files: {file_path: {"collection": str, "last_event": float}}
        self._pending: dict[str, dict] = {}
        self._lock = threading.Lock()

        # Files currently being indexed (prevent re-entry)
        self._indexing: set[str] = set()
        self._indexing_lock = threading.Lock()

        # Flush thread
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()

        # VaultMemory instance (created lazily per flush to avoid long-lived connections)
        self._vm: Optional[VaultMemory] = None

    def enqueue(self, file_path: str, collection: str):
        """Record a file change. Actual indexing happens after debounce."""
        with self._lock:
            self._pending[file_path] = {
                "collection": collection,
                "last_event": time.time(),
            }
            pending_count = len(self._pending)

        if pending_count >= MAX_BATCH_SIZE:
            logger.info(f"Batch size {pending_count} >= {MAX_BATCH_SIZE}, forcing flush")

    def _flush_loop(self):
        """Background thread: periodically check for debounced files and index them."""
        while self._running:
            time.sleep(FLUSH_CHECK_INTERVAL)
            self._try_flush()

    def _try_flush(self):
        """Find files whose debounce window has expired and index them."""
        now = time.time()
        ready: dict[str, dict] = {}

        with self._lock:
            still_pending: dict[str, dict] = {}
            for fp, info in self._pending.items():
                age = now - info["last_event"]
                if age >= self._debounce_s:
                    ready[fp] = info
                else:
                    still_pending[fp] = info
            self._pending = still_pending

        if not ready:
            return

        # Filter out files currently being indexed
        with self._indexing_lock:
            to_index = {fp: info for fp, info in ready.items() if fp not in self._indexing}
            if not to_index:
                return
            for fp in to_index:
                self._indexing.add(fp)

        logger.info(f"Flushing {len(to_index)} files for indexing")

        try:
            vm = self._get_vm()

            for fp, info in to_index.items():
                self._index_one(vm, fp, info["collection"])
        except Exception as e:
            logger.error(f"Flush error: {e}")
        finally:
            with self._indexing_lock:
                for fp in to_index:
                    self._indexing.discard(fp)

    def _index_one(self, vm: VaultMemory, file_path: str, collection: str):
        """Index a single file. Never raises."""
        try:
            if not os.path.exists(file_path):
                # File was deleted — prune from DB
                logger.info(f"Pruning deleted file: {file_path}")
                vm._store_().prune_file(file_path)
                return

            result = vm.index_file(file_path, collection=collection)
            if result.skipped:
                logger.debug(f"Skipped (unchanged): {file_path}")
            else:
                logger.info(
                    f"Indexed: {file_path} "
                    f"(chunks={result.chunks_added}, embedded={result.embedded})"
                )
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")

    def _get_vm(self) -> VaultMemory:
        """Get or create VaultMemory instance."""
        if self._vm is None:
            self._vm = VaultMemory(self._db_path, nvidia_api_key=self._nvidia_key)
        return self._vm

    def stop(self):
        """Graceful shutdown: flush remaining, close DB."""
        self._running = False
        self._flush_thread.join(timeout=10)

        # Final flush
        with self._lock:
            remaining = dict(self._pending)
            self._pending.clear()

        if remaining:
            logger.info(f"Final flush: {len(remaining)} files")
            try:
                vm = self._get_vm()
                for fp, info in remaining.items():
                    self._index_one(vm, fp, info["collection"])
            except Exception as e:
                logger.error(f"Final flush error: {e}")

        if self._vm:
            self._vm.close()
            self._vm = None

        logger.info("Indexer stopped")


# ── File System Event Handler ──────────────────────────────────────────────────

class VaultEventHandler(FileSystemEventHandler):
    """Watches for .md file changes and enqueues them for indexing."""

    def __init__(self, collection: str, indexer: DebouncedIndexer):
        super().__init__()
        self.collection = collection
        self.indexer = indexer

    def _should_process(self, path: str) -> bool:
        """Filter: only .md files, skip hidden dirs and temp files."""
        p = Path(path)

        # Only markdown
        if p.suffix.lower() != ".md":
            return False

        # Skip hidden directories (.obsidian, .git, .trash, etc.)
        parts = p.parts
        for part in parts:
            if part.startswith(".") and part not in (".", ".."):
                return False

        # Skip temp files (editors often create these)
        name = p.name
        if name.startswith(".") or name.startswith("~") or name.endswith(".tmp"):
            return False

        return True

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process(event.src_path):
            logger.debug(f"Created: {event.src_path}")
            self.indexer.enqueue(event.src_path, self.collection)

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process(event.src_path):
            logger.debug(f"Modified: {event.src_path}")
            self.indexer.enqueue(event.src_path, self.collection)

    def on_moved(self, event: FileSystemEvent):
        # Old path: prune. New path: index.
        if not event.is_directory:
            if self._should_process(event.src_path):
                logger.debug(f"Moved from: {event.src_path}")
                self.indexer.enqueue(event.src_path, self.collection)
            if self._should_process(event.dest_path):
                logger.debug(f"Moved to: {event.dest_path}")
                self.indexer.enqueue(event.dest_path, self.collection)

    def on_deleted(self, event: FileSystemEvent):
        if not event.is_directory and self._should_process(event.src_path):
            logger.debug(f"Deleted: {event.src_path}")
            self.indexer.enqueue(event.src_path, self.collection)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    if not NVIDIA_KEY:
        logger.error("NVIDIA_API_KEY not set. Cannot generate embeddings.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("vault-watcher starting")
    logger.info(f"  DB: {DEFAULT_DB}")
    logger.info(f"  Debounce: {DEBOUNCE_SECONDS}s")
    logger.info(f"  Max batch: {MAX_BATCH_SIZE}")

    indexer = DebouncedIndexer(DEFAULT_DB, NVIDIA_KEY, DEBOUNCE_SECONDS)
    observer = Observer()

    for watch_dir in WATCH_DIRS:
        path = watch_dir["path"]
        collection = watch_dir["collection"]

        if not os.path.isdir(path):
            logger.warning(f"  Watch dir not found, skipping: {path}")
            continue

        handler = VaultEventHandler(collection, indexer)
        observer.schedule(handler, path, recursive=True)
        logger.info(f"  Watching: {path} → [{collection}]")

    observer.start()
    logger.info("Watcher active. Waiting for file changes...")

    # Graceful shutdown
    shutdown_event = threading.Event()

    def _signal_handler(signum, frame):
        signame = signal.Signals(signum).name
        logger.info(f"Received {signame}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        pass

    logger.info("Stopping observer...")
    observer.stop()
    observer.join(timeout=10)

    logger.info("Stopping indexer...")
    indexer.stop()

    logger.info("vault-watcher stopped")


if __name__ == "__main__":
    main()
