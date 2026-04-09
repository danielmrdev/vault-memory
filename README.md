# vault-memory

Semantic search over collections of markdown files.

BM25 (FTS5) + vector cosine similarity, combined via Reciprocal Rank Fusion.
SQLite-backed, zero external dependencies, free Gemini embeddings (free tier).

Ported and extracted from [Nous](https://github.com/danielmunoz/nous) —
a personal AI assistant. Designed to work standalone or as a
[Hermes Agent](https://hermes.computer) memory plugin.

---

## Features

- **Hybrid search** — BM25 keyword + cosine vector similarity, fused via RRF
- **Agnostic collections** — index any directory under any name (`obsidian`, `hermes-memory`, `work-notes`, …)
- **Incremental indexing** — skips unchanged files (mtime), re-embeds only new/changed chunks
- **Graceful degradation** — falls back to BM25-only when Gemini is unavailable (rate limit, no key)
- **Zero dependencies** — Python stdlib only (`sqlite3`, `struct`, `urllib`)
- **Compatible DB** — same schema as Nous `vectors.db`, can reuse existing data

---

## Installation

### As a standalone CLI

```bash
git clone https://github.com/YOUR_USER/vault-memory
cd vault-memory
pip install -e .          # installs the `vm` command

# or without installing, run directly:
python -m vault_memory.cli
```

### As a Hermes Agent memory plugin

```bash
# 1. Clone into Hermes plugin directory
git clone https://github.com/YOUR_USER/vault-memory \
    ~/.hermes/hermes-agent/plugins/memory/vault-memory

# 2. Add GEMINI_API_KEY to ~/.hermes/.env
echo "GEMINI_API_KEY=your_key_here" >> ~/.hermes/.env

# 3. Set default DB path (optional — default: ~/hache/data/vectors.db)
echo "VAULT_MEMORY_DB=/path/to/vectors.db" >> ~/.hermes/.env

# 4. Enable in Hermes
hermes memory setup
# → select: vault-memory

# 5. Index your vault
vm index ~/path/to/obsidian-vault --collection obsidian
vm index ~/.hermes/memories --collection hermes-memory
```

### Scheduled reindexing (cron)

A ready-made script for nightly reindexing with weekly rotation to respect
the Gemini free tier quota:

```bash
# Add to crontab — runs at 4am every day
crontab -e
# Add: 0 4 * * * /path/to/hache/scripts/vector-reindex.sh >> ~/hache/logs/vector-reindex.log 2>&1
```

---

## Usage

### CLI (`vm`)

```bash
# Index a directory (any name for collection)
vm index ~/obsidian-vault         --collection obsidian
vm index ~/.hermes/memories       --collection hermes-memory
vm index ~/notes/work             --collection work

# Index a single file
vm index ~/obsidian-vault/note.md --collection obsidian

# Search (hybrid by default — BM25+vector if embeddings exist, else BM25-only)
vm search "pitchgale marketing strategy" --collection obsidian
vm search "how to configure vllm"        --limit 10
vm search "weekly review"                --json    # JSON output for scripting

# Stats
vm stats

# Prune orphaned entries (files that were deleted)
vm prune ~/obsidian-vault
```

### Environment variables

| Variable         | Default                       | Description                     |
|------------------|-------------------------------|---------------------------------|
| `VAULT_MEMORY_DB`| `~/hache/data/vectors.db`     | Path to SQLite database         |
| `GEMINI_API_KEY` | _(empty — BM25 only)_         | Gemini API key for embeddings   |

### Python API

```python
from vault_memory import VaultMemory

vm = VaultMemory("/path/to/vectors.db", gemini_api_key="...")

# Index
vm.index_directory("~/obsidian-vault", collection="obsidian")
vm.index_file("~/notes/project.md",    collection="work")

# Search
results = vm.search("my query", collection="obsidian", limit=5)

# Format for LLM context injection
context = vm.build_context(results)
# → "## Relevant Context\n\n**heading** (file.md)\nsnippet...\n\n..."

# Stats
stats = vm.get_stats()

# Prune deleted files
vm.prune_orphans("~/obsidian-vault")

vm.close()
```

---

## Architecture

```
vault_memory/
├── chunker.py    — Markdown → overlapping chunks (1500 chars, 200 overlap)
├── store.py      — SQLite: FTS5 BM25 + BLOB embeddings + RRF combiner
├── embedder.py   — Gemini API client (batching, retry, rate-limit cooldown)
├── indexer.py    — High-level facade: index_file, index_directory, search
└── cli.py        — CLI entry point (vm command)
```

### DB Schema

```sql
collections   -- named collections
chunks        -- text chunks (content, heading, embedding BLOB, norm)
ingest_log    -- per-file mtime for skip-on-unchanged
chunks_fts    -- FTS5 virtual table (porter tokenizer) with auto-sync triggers
```

### Search modes

| Mode     | When                                               |
|----------|----------------------------------------------------|
| `hybrid` | Gemini available + collection has embeddings       |
| `bm25`   | Gemini unavailable / rate limited / no embeddings  |

Hybrid uses RRF (k=60) to merge vector and BM25 ranked lists.

---

## Gemini free tier

`gemini-embedding-001` free tier: ~1500 RPD (requests per day), 1500 RPM.

The nightly reindex script uses weekly rotation to stay within quota:
- Mon: `00-Inbox`
- Tue: `01-PROYECTOS`
- Wed: `02-ÁREAS`
- Thu: `03-RECURSOS`
- Fri: `04-Archivo`
- Sat/Sun: retry unembedded chunks only

---

## License

MIT
