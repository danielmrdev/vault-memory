"""
VaultMemory — high-level facade for indexing and searching.

Collections are just (name, path_to_index) pairs — no hardcoded obsidian/nous.
The DB is shared across all collections.

Usage:
    vm = VaultMemory("/path/to/vectors.db", gemini_api_key="...")
    vm.index_file("/path/to/note.md", collection="obsidian")
    vm.index_directory("/path/to/vault", collection="obsidian")
    results = vm.search("my query", collection="obsidian")
    print(vm.build_context(results))
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .chunker import chunk_markdown
from .embedder import GeminiEmbedder
from .store import SearchResult, VectorStore

logger = logging.getLogger(__name__)

SEMANTIC_CONTEXT_MAX_CHARS = 4000
RESULT_MAX_CHARS = 1000


@dataclass
class FileIndexResult:
    file: str
    chunks_added: int
    embedded: int
    skipped: bool


@dataclass
class IndexSummary:
    files_processed: int = 0
    chunks_added: int = 0
    embedded: int = 0
    skipped: int = 0
    errors: list[dict] = field(default_factory=list)
    elapsed_ms: int = 0


class VaultMemory:
    """Facade — lazy-initialised store + embedder, safe on the hot search path."""

    def __init__(self, db_path: str, gemini_api_key: str = ""):
        self._db_path = db_path
        self._api_key = gemini_api_key
        self._store: Optional[VectorStore] = None
        self._embedder: Optional[GeminiEmbedder] = None

    def _store_(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(self._db_path)
        return self._store

    def _embedder_(self) -> GeminiEmbedder:
        if self._embedder is None:
            self._embedder = GeminiEmbedder(self._api_key)
        return self._embedder

    def close(self):
        if self._store:
            self._store.close()
            self._store = None

    # ── Search ────────────────────────────────────────────────────────────

    def search(self, query: str, collection: Optional[str] = None,
               limit: int = 5) -> list[SearchResult]:
        """Hybrid (BM25 + vector) or BM25-only search. Never raises."""
        try:
            store = self._store_()
            embedder = self._embedder_()
            use_vector = (
                embedder.is_available
                and store.has_embeddings(collection)
            )

            if use_vector:
                vecs = embedder.embed_batch([query])
                if vecs:
                    logger.debug(f'[VaultMemory] search mode=hybrid query="{query[:50]}"')
                    return store.search_hybrid(query, vecs[0], collection=collection, limit=limit)
                logger.debug(f'[VaultMemory] search mode=bm25-fallback (embed failed)')

            logger.debug(f'[VaultMemory] search mode=bm25-only query="{query[:50]}"')
            return store.search_bm25(query, collection=collection, limit=limit)
        except Exception as e:
            logger.error(f"[VaultMemory] search error: {e}")
            return []

    def build_context(self, results: list[SearchResult]) -> str:
        """Format results into a ## Relevant Context block for the system prompt."""
        if not results:
            return ""

        seen: set[str] = set()
        context = "## Relevant Context\n\n"
        remaining = SEMANTIC_CONTEXT_MAX_CHARS - len(context)

        for r in results:
            if r.file_path in seen:
                continue
            seen.add(r.file_path)

            snippet = r.content[:RESULT_MAX_CHARS] + ("..." if len(r.content) > RESULT_MAX_CHARS else "")
            title = r.heading or Path(r.file_path).name
            entry = f"**{title}** ({r.file_path})\n{snippet}\n\n"

            if len(entry) > remaining:
                break
            context += entry
            remaining -= len(entry)

        return context

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_file(self, file_path: str,
                   collection: str = "default") -> FileIndexResult:
        """Index a single markdown file.

        Steps:
          1. stat → mtime_ms
          2. Check ingest_log → skip if unchanged (unless unembedded chunks exist)
          3. Read + chunk
          4. Upsert chunks
          5. Embed unembedded chunks (if Gemini available)
        """
        store = self._store_()
        store.ensure_collection(collection)

        stat = os.stat(file_path)
        mtime_ms = int(stat.st_mtime * 1000)

        # Check if we can skip
        existing = store.get_ingest_status(file_path)
        if existing and existing["mtime"] == mtime_ms:
            # File unchanged — retry pending embeddings if possible
            embedder = self._embedder_()
            if embedder.is_available:
                pending = store.get_unembedded_chunks(file_path)
                if pending:
                    texts = [c["content"] for c in pending]
                    embeddings = embedder.embed_batch(texts)
                    if embeddings:
                        count = min(len(pending), len(embeddings))
                        store.update_embeddings([
                            {"id": pending[i]["id"], "embedding": embeddings[i]}
                            for i in range(count)
                        ])
                        logger.info(f'[VaultMemory] embedding-retry file="{file_path}" embedded={count}')
                        return FileIndexResult(file=file_path, chunks_added=0,
                                               embedded=count, skipped=False)
            return FileIndexResult(file=file_path, chunks_added=0,
                                   embedded=0, skipped=True)

        # Read and chunk
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        chunks = chunk_markdown(file_path, content, mtime_ms)

        if not chunks:
            store.update_ingest_log(file_path, collection, mtime_ms, 0)
            return FileIndexResult(file=file_path, chunks_added=0,
                                   embedded=0, skipped=False)

        store.upsert_chunks(chunks, collection)
        store.update_ingest_log(file_path, collection, mtime_ms, len(chunks))

        # Embed unembedded chunks
        embedded = 0
        embedder = self._embedder_()
        if embedder.is_available:
            try:
                pending = store.get_unembedded_chunks(file_path)
                if pending:
                    texts = [c["content"] for c in pending]
                    embeddings = embedder.embed_batch(texts)
                    if embeddings:
                        count = min(len(pending), len(embeddings))
                        store.update_embeddings([
                            {"id": pending[i]["id"], "embedding": embeddings[i]}
                            for i in range(count)
                        ])
                        embedded = count
            except Exception as e:
                logger.warning(f"[VaultMemory] embedding failed for {file_path}: {e}")

        logger.info(f'[VaultMemory] indexed file="{file_path}" chunks={len(chunks)} embedded={embedded}')
        return FileIndexResult(file=file_path, chunks_added=len(chunks),
                               embedded=embedded, skipped=False)

    def index_directory(self, dir_path: str,
                        collection: str = "default",
                        glob: str = "**/*.md") -> IndexSummary:
        """Index all matching files under dir_path.

        collection — any name, e.g. "obsidian", "hermes-sessions", "skills"
        glob       — default "**/*.md", change for other file types
        """
        import time
        start = time.time()
        summary = IndexSummary()

        paths = list(Path(dir_path).glob(glob))
        logger.info(f'[VaultMemory] indexDirectory dir="{dir_path}" collection="{collection}" files={len(paths)}')

        for fp in paths:
            try:
                result = self.index_file(str(fp), collection=collection)
                summary.files_processed += 1
                if result.skipped:
                    summary.skipped += 1
                else:
                    summary.chunks_added += result.chunks_added
                    summary.embedded += result.embedded
            except Exception as e:
                summary.errors.append({"file": str(fp), "error": str(e)})

        summary.elapsed_ms = int((time.time() - start) * 1000)
        logger.info(
            f'[VaultMemory] done dir="{dir_path}" files={summary.files_processed} '
            f'chunks={summary.chunks_added} embedded={summary.embedded} '
            f'skipped={summary.skipped} errors={len(summary.errors)} '
            f'elapsed={summary.elapsed_ms}ms'
        )
        return summary

    def prune_orphans(self, dir_path: str) -> list[str]:
        """Remove ingest_log entries for files that no longer exist."""
        store = self._store_()
        indexed = store.get_indexed_files(dir_path)
        pruned = []
        for fp in indexed:
            if not Path(fp).exists():
                store.prune_file(fp)
                pruned.append(fp)
                logger.info(f'[VaultMemory] pruned orphan "{fp}"')
        return pruned

    def get_stats(self):
        return self._store_().get_stats()
