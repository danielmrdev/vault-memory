"""
VectorStore — SQLite-backed storage for vault-memory.

Compatible with the Nous vectors.db schema so existing data is reused
without migration. Python port of Nous vector-memory/store.ts.

Schema:
  collections   — named collections (e.g. "obsidian", "hermes-memory")
  chunks        — text chunks with optional BLOB embeddings + L2 norm
  ingest_log    — per-file mtime + chunk count for skip-on-unchanged
  chunks_fts    — FTS5 virtual table (porter tokeniser) with AI/AU/AD triggers
"""

from __future__ import annotations

import math
import re
import sqlite3
import struct
from pathlib import Path
from typing import Optional

from .chunker import ChunkData


# ── Schema ────────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS collections (
    id          TEXT PRIMARY KEY,
    description TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    collection_id TEXT    NOT NULL REFERENCES collections(id),
    file_path     TEXT    NOT NULL,
    file_mtime    INTEGER NOT NULL,
    chunk_index   INTEGER NOT NULL,
    content       TEXT    NOT NULL,
    heading       TEXT,
    token_count   INTEGER NOT NULL DEFAULT 0,
    norm          REAL,
    embedding     BLOB,
    created_at    TEXT    NOT NULL DEFAULT (datetime('now')),
    UNIQUE(file_path, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_file
    ON chunks(file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_collection
    ON chunks(collection_id);
CREATE INDEX IF NOT EXISTS idx_chunks_has_embedding
    ON chunks(collection_id) WHERE embedding IS NOT NULL;

CREATE TABLE IF NOT EXISTS ingest_log (
    file_path       TEXT PRIMARY KEY,
    collection_id   TEXT    NOT NULL REFERENCES collections(id),
    mtime           INTEGER NOT NULL,
    chunk_count     INTEGER NOT NULL DEFAULT 0,
    embedded_count  INTEGER NOT NULL DEFAULT 0,
    last_indexed    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    tokenize='porter ascii'
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(chunk_id, content) VALUES (NEW.id, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.id;
    INSERT INTO chunks_fts(chunk_id, content) VALUES (NEW.id, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    DELETE FROM chunks_fts WHERE chunk_id = OLD.id;
END;
"""

DEFAULT_COLLECTIONS = [
    ("obsidian",      "Obsidian vault notes"),
    ("hermes-memory", "Hermes sessions, skills, and memory files"),
]


# ── Data classes ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class SearchResult:
    chunk_id: int
    file_path: str
    content: str
    heading: Optional[str]
    score: float
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None


@dataclass
class IndexStats:
    collections: list[dict]
    total_chunks: int
    total_embedded: int
    db_path: str


# ── VectorStore ───────────────────────────────────────────────────────────────

class VectorStore:
    """SQLite-backed chunk store. Thread-safe for single-writer use (WAL mode)."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # check_same_thread=False safe here: single writer + WAL reads
        self._con = sqlite3.connect(db_path, check_same_thread=False)
        self._con.row_factory = sqlite3.Row

        if db_path != ":memory:":
            self._con.execute("PRAGMA journal_mode=WAL")

        self._init_schema()

    def close(self):
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Schema ────────────────────────────────────────────────────────────

    def _init_schema(self):
        self._con.executescript(SCHEMA_SQL)
        with self._con:
            for cid, desc in DEFAULT_COLLECTIONS:
                self._con.execute(
                    "INSERT OR IGNORE INTO collections (id, description) VALUES (?, ?)",
                    (cid, desc),
                )

    def ensure_collection(self, collection_id: str, description: str = ""):
        """Create a collection if it doesn't exist yet."""
        with self._con:
            self._con.execute(
                "INSERT OR IGNORE INTO collections (id, description) VALUES (?, ?)",
                (collection_id, description or collection_id),
            )

    # ── Chunk CRUD ────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[ChunkData], collection_id: str) -> list[int]:
        """Insert or update chunks.

        ON CONFLICT preserves the existing embedding when content is unchanged,
        clears it when content changed (forces re-embedding).

        Returns list of row IDs (lastrowid for inserts, existing id for updates).
        """
        sql = """
            INSERT INTO chunks
                (collection_id, file_path, file_mtime, chunk_index, content,
                 heading, token_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ON CONFLICT(file_path, chunk_index) DO UPDATE SET
                collection_id = excluded.collection_id,
                file_mtime    = excluded.file_mtime,
                content       = excluded.content,
                heading       = excluded.heading,
                token_count   = excluded.token_count,
                embedding     = CASE WHEN excluded.content != chunks.content
                                    THEN NULL ELSE chunks.embedding END,
                norm          = CASE WHEN excluded.content != chunks.content
                                    THEN NULL ELSE chunks.norm END
        """
        ids = []
        with self._con:
            for c in chunks:
                cur = self._con.execute(sql, (
                    collection_id, c.file_path, c.file_mtime, c.chunk_index,
                    c.content, c.heading, c.token_count,
                ))
                # lastrowid is the inserted row for INSERT, for UPDATE we need
                # to look it up by (file_path, chunk_index)
                if cur.lastrowid:
                    ids.append(cur.lastrowid)
                else:
                    row = self._con.execute(
                        "SELECT id FROM chunks WHERE file_path=? AND chunk_index=?",
                        (c.file_path, c.chunk_index),
                    ).fetchone()
                    ids.append(row["id"] if row else -1)
        return ids

    def get_unembedded_chunks(self, file_path: str) -> list[dict]:
        """Return chunks for a file that have no embedding yet."""
        rows = self._con.execute(
            "SELECT id, content FROM chunks WHERE file_path=? AND embedding IS NULL "
            "ORDER BY chunk_index",
            (file_path,),
        ).fetchall()
        return [{"id": r["id"], "content": r["content"]} for r in rows]

    def get_pending_embeddings(self, collection_id: Optional[str] = None,
                              limit: int = 100) -> list[dict]:
        """Return unembedded chunks for a collection (for daily embedding jobs)."""
        if collection_id:
            rows = self._con.execute(
                "SELECT id, content FROM chunks "
                "WHERE collection_id=? AND embedding IS NULL "
                "ORDER BY file_path, chunk_index "
                "LIMIT ?",
                (collection_id, limit),
            ).fetchall()
        else:
            rows = self._con.execute(
                "SELECT id, content FROM chunks "
                "WHERE embedding IS NULL "
                "ORDER BY collection_id, file_path, chunk_index "
                "LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"id": r["id"], "content": r["content"]} for r in rows]

    def update_embeddings(self, items: list[dict]):
        """Write embeddings (list[float]) and precomputed L2 norms to chunks.

        items: [{"id": int, "embedding": list[float]}, ...]
        """
        sql = "UPDATE chunks SET embedding=?, norm=? WHERE id=?"
        with self._con:
            for item in items:
                emb = item["embedding"]
                blob = _floats_to_blob(emb)
                norm = _l2_norm(emb)
                self._con.execute(sql, (blob, norm, item["id"]))

    # ── Ingest log ────────────────────────────────────────────────────────

    def update_ingest_log(self, file_path: str, collection_id: str,
                          mtime: int, chunk_count: int):
        with self._con:
            self._con.execute("""
                INSERT INTO ingest_log
                    (file_path, collection_id, mtime, chunk_count, embedded_count, last_indexed)
                VALUES (?, ?, ?, ?, 0, datetime('now'))
                ON CONFLICT(file_path) DO UPDATE SET
                    collection_id = excluded.collection_id,
                    mtime         = excluded.mtime,
                    chunk_count   = excluded.chunk_count,
                    last_indexed  = excluded.last_indexed
            """, (file_path, collection_id, mtime, chunk_count))

    def get_ingest_status(self, file_path: str) -> Optional[dict]:
        """Return {"mtime": int, "chunk_count": int} or None."""
        row = self._con.execute(
            "SELECT mtime, chunk_count FROM ingest_log WHERE file_path=?",
            (file_path,),
        ).fetchone()
        if not row:
            return None
        return {"mtime": row["mtime"], "chunk_count": row["chunk_count"]}

    def prune_file(self, file_path: str):
        """Delete all chunks and ingest log entry for a file."""
        with self._con:
            self._con.execute("DELETE FROM chunks WHERE file_path=?", (file_path,))
            self._con.execute("DELETE FROM ingest_log WHERE file_path=?", (file_path,))

    def get_indexed_files(self, dir_path: str) -> list[str]:
        """Return all file paths in ingest_log under dir_path."""
        prefix = dir_path.rstrip("/") + "/"
        rows = self._con.execute(
            "SELECT file_path FROM ingest_log WHERE file_path LIKE ? ESCAPE '\\'",
            (prefix.replace("%", "\\%").replace("_", "\\_") + "%",),
        ).fetchall()
        return [r["file_path"] for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> IndexStats:
        rows = self._con.execute("""
            SELECT
                c.id,
                c.description,
                COUNT(ch.id)              AS chunk_count,
                COUNT(ch.embedding)       AS embedded_count,
                COUNT(DISTINCT ch.file_path) AS file_count
            FROM collections c
            LEFT JOIN chunks ch ON ch.collection_id = c.id
            GROUP BY c.id, c.description
            ORDER BY c.id
        """).fetchall()

        collections = [
            {
                "id": r["id"],
                "description": r["description"] or "",
                "chunk_count": r["chunk_count"],
                "embedded_count": r["embedded_count"],
                "file_count": r["file_count"],
            }
            for r in rows
        ]
        total_chunks   = sum(c["chunk_count"]   for c in collections)
        total_embedded = sum(c["embedded_count"] for c in collections)

        return IndexStats(
            collections=collections,
            total_chunks=total_chunks,
            total_embedded=total_embedded,
            db_path=self.db_path,
        )

    def has_embeddings(self, collection_id: Optional[str] = None) -> bool:
        sql = "SELECT 1 FROM chunks WHERE embedding IS NOT NULL"
        params = []
        if collection_id:
            sql += " AND collection_id=?"
            params.append(collection_id)
        sql += " LIMIT 1"
        return self._con.execute(sql, params).fetchone() is not None

    # ── Search ────────────────────────────────────────────────────────────

    def search_bm25(self, query: str, collection: Optional[str] = None,
                    limit: int = 5) -> list[SearchResult]:
        """Full-text BM25 search via FTS5 porter tokeniser."""
        sanitized = _sanitize_fts(query)
        if not sanitized:
            return []

        if collection:
            sql = """
                SELECT c.id, c.file_path, c.content, c.heading,
                       bm25(chunks_fts) AS bm25_score
                FROM chunks c
                JOIN chunks_fts f ON f.chunk_id = c.id
                WHERE chunks_fts MATCH ? AND c.collection_id = ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """
            rows = self._con.execute(sql, (sanitized, collection, limit)).fetchall()
        else:
            sql = """
                SELECT c.id, c.file_path, c.content, c.heading,
                       bm25(chunks_fts) AS bm25_score
                FROM chunks c
                JOIN chunks_fts f ON f.chunk_id = c.id
                WHERE chunks_fts MATCH ?
                ORDER BY bm25(chunks_fts)
                LIMIT ?
            """
            rows = self._con.execute(sql, (sanitized, limit)).fetchall()

        return [
            SearchResult(
                chunk_id=r["id"],
                file_path=r["file_path"],
                content=r["content"],
                heading=r["heading"],
                score=-r["bm25_score"] / 10,
                bm25_score=r["bm25_score"],
            )
            for r in rows
        ]

    def search_vector(self, query_embedding: list[float],
                      collection: Optional[str] = None,
                      limit: int = 5) -> list[SearchResult]:
        """Cosine similarity search over embedded chunks."""
        q_norm = _l2_norm(query_embedding)
        if q_norm == 0:
            return []

        # Normalise query vector
        q = [v / q_norm for v in query_embedding]

        if collection:
            rows = self._con.execute(
                "SELECT id, file_path, content, heading, embedding, norm "
                "FROM chunks WHERE embedding IS NOT NULL AND collection_id=?",
                (collection,),
            ).fetchall()
        else:
            rows = self._con.execute(
                "SELECT id, file_path, content, heading, embedding, norm "
                "FROM chunks WHERE embedding IS NOT NULL",
            ).fetchall()

        if not rows:
            return []

        scored = []
        for r in rows:
            stored = _blob_to_floats(r["embedding"])
            s_norm = r["norm"] or _l2_norm(stored)
            if s_norm == 0:
                continue
            dot = sum(q[i] * (stored[i] / s_norm) for i in range(min(len(q), len(stored))))
            scored.append((dot, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(
                chunk_id=r["id"],
                file_path=r["file_path"],
                content=r["content"],
                heading=r["heading"],
                score=dot,
                vector_score=dot,
            )
            for dot, r in scored[:limit]
        ]

    def search_hybrid(self, query: str, query_embedding: list[float],
                      collection: Optional[str] = None,
                      limit: int = 5) -> list[SearchResult]:
        """BM25 + vector combined via Reciprocal Rank Fusion."""
        bm25 = self.search_bm25(query, collection=collection, limit=limit)
        vec  = self.search_vector(query_embedding, collection=collection, limit=limit)
        return _rrf_combine(vec, bm25)[:limit]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _floats_to_blob(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)


def _blob_to_floats(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _sanitize_fts(query: str) -> str:
    return re.sub(r"[^\w\s]", " ", query).strip()


def _rrf_combine(vector_results: list[SearchResult],
                 bm25_results: list[SearchResult],
                 k: int = 60) -> list[SearchResult]:
    """Reciprocal Rank Fusion — Cormack et al. 2009, k=60."""
    scores: dict[int, dict] = {}

    for rank, r in enumerate(vector_results):
        scores[r.chunk_id] = {
            "result": r, "score": 1 / (k + rank),
            "vector_score": r.vector_score or r.score, "bm25_score": None,
        }

    for rank, r in enumerate(bm25_results):
        if r.chunk_id in scores:
            scores[r.chunk_id]["score"] += 1 / (k + rank)
            scores[r.chunk_id]["bm25_score"] = r.bm25_score
        else:
            scores[r.chunk_id] = {
                "result": r, "score": 1 / (k + rank),
                "vector_score": None, "bm25_score": r.bm25_score,
            }

    combined = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
    return [
        SearchResult(
            chunk_id=e["result"].chunk_id,
            file_path=e["result"].file_path,
            content=e["result"].content,
            heading=e["result"].heading,
            score=e["score"],
            vector_score=e["vector_score"],
            bm25_score=e["bm25_score"],
        )
        for e in combined
    ]
