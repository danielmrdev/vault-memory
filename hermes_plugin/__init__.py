"""
vault-memory Hermes Plugin — MemoryProvider implementation.

Provides semantic search over indexed collections (Obsidian vault,
Hermes sessions, skills, etc.) via the MemoryProvider ABC.

Hooks used:
  prefetch()          — hybrid search before each turn, injects context
  get_tool_schemas()  — exposes vault_search tool to the agent
  handle_tool_call()  — executes vault_search
  system_prompt_block() — brief status line
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Try to import the MemoryProvider ABC ─────────────────────────────────────
try:
    from agent.memory_provider import MemoryProvider
except ImportError:
    # Fallback for standalone testing outside Hermes
    from abc import ABC, abstractmethod
    class MemoryProvider(ABC):  # type: ignore
        @property
        @abstractmethod
        def name(self) -> str: ...
        @abstractmethod
        def is_available(self) -> bool: ...
        @abstractmethod
        def initialize(self, session_id: str, **kwargs) -> None: ...
        @abstractmethod
        def get_tool_schemas(self) -> List[Dict[str, Any]]: ...


# ── Tool schema ───────────────────────────────────────────────────────────────

VAULT_SEARCH_SCHEMA = {
    "name": "vault_search",
    "description": (
        "Search the semantic knowledge base — covers the Obsidian vault "
        "(notes, projects, areas, resources) and Hermes memory files "
        "(sessions, skills, memories). "
        "Uses hybrid BM25 + vector search (NVIDIA embeddings). "
        "Use this when you need context about Daniel's projects, past decisions, "
        "notes, or anything that might be in the knowledge base."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural language search query.",
            },
            "collection": {
                "type": "string",
                "description": (
                    "Optional collection to search. "
                    "Leave empty to search all. "
                    "Known collections: obsidian, hermes-exported."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max results to return (default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


# ── Provider ──────────────────────────────────────────────────────────────────

class VaultMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider backed by vault-memory."""

    def __init__(self):
        self._vm = None          # VaultMemory instance, lazy
        self._prefetch_cache: Optional[str] = None
        self._prefetch_lock = threading.Lock()
        self._db_path: str = ""
        self._api_key: str = ""

    # ── MemoryProvider ABC ────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "vault-memory"

    def is_available(self) -> bool:
        db = self._resolve_db_path()
        return bool(db)  # available even without NVIDIA key (BM25-only mode)

    def initialize(self, session_id: str, **kwargs) -> None:
        self._db_path = self._resolve_db_path()
        self._api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not self._db_path:
            logger.warning("[VaultMemory] VAULT_MEMORY_DB not set and default not found")
            return
        self._get_vm()  # open connection eagerly
        logger.info(f"[VaultMemory] initialized db={self._db_path}")

    def system_prompt_block(self) -> str:
        if not self._vm:
            return ""
        try:
            stats = self._vm.get_stats()
            total = stats.total_chunks
            embedded = stats.total_embedded
            mode = "hybrid" if embedded > 0 else "BM25-only"
            return (
                f"Semantic knowledge base active — "
                f"{total} chunks across {len(stats.collections)} collections "
                f"({mode}). Use vault_search to retrieve relevant context."
            )
        except Exception:
            return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        with self._prefetch_lock:
            result = self._prefetch_cache
            self._prefetch_cache = None
        if result is not None:
            return result
        # Synchronous fallback if queue_prefetch wasn't called
        return self._do_search(query)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        def _run():
            result = self._do_search(query)
            with self._prefetch_lock:
                self._prefetch_cache = result
        threading.Thread(target=_run, daemon=True).start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [VAULT_SEARCH_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name != "vault_search":
            raise NotImplementedError(f"Unknown tool: {tool_name}")

        query = args.get("query", "")
        collection = args.get("collection") or None
        limit = int(args.get("limit", 5))

        if not query:
            return json.dumps({"error": "query is required"})

        vm = self._get_vm()
        if not vm:
            return json.dumps({"error": "vault-memory not initialised"})

        results = vm.search(query, collection=collection, limit=limit)
        return json.dumps(
            [
                {
                    "file": r.file_path,
                    "heading": r.heading,
                    "score": round(r.score, 4),
                    "content": r.content[:800],
                }
                for r in results
            ],
            ensure_ascii=False,
        )

    def shutdown(self) -> None:
        if self._vm:
            self._vm.close()
            self._vm = None

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {
                "key": "db_path",
                "description": "Path to vault-memory SQLite database",
                "required": False,
                "default": "~/hache/data/vectors.db",
                "env_var": "VAULT_MEMORY_DB",
                "secret": False,
            },
            {
                "key": "nvidia_api_key",
                "description": "NVIDIA NIM API key for vector embeddings (free tier, model: nv-embedqa-e5-v5). Leave empty for BM25-only.",
                "required": False,
                "env_var": "NVIDIA_API_KEY",
                "secret": True,
                "url": "https://build.nvidia.com/",
            },
        ]

    # ── Internal helpers ──────────────────────────────────────────────────

    def _resolve_db_path(self) -> str:
        path = os.environ.get(
            "VAULT_MEMORY_DB",
            os.path.expanduser("~/hache/data/vectors.db"),
        )
        return os.path.expanduser(path)

    def _get_vm(self):
        if self._vm is None and self._db_path:
            # Import lazily so the plugin loads even if vault_memory isn't installed
            try:
                import sys
                plugin_dir = Path(__file__).parent.parent
                if str(plugin_dir) not in sys.path:
                    sys.path.insert(0, str(plugin_dir))
                from vault_memory import VaultMemory
                self._vm = VaultMemory(self._db_path, nvidia_api_key=self._api_key)
            except ImportError as e:
                logger.error(f"[VaultMemory] Failed to import vault_memory: {e}")
        return self._vm

    def _do_search(self, query: str, limit: int = 5) -> str:
        vm = self._get_vm()
        if not vm or not query.strip():
            return ""
        try:
            results = vm.search(query, limit=limit)
            return vm.build_context(results)
        except Exception as e:
            logger.error(f"[VaultMemory] prefetch error: {e}")
            return ""


# Hermes discovers this via the module-level `provider` attribute
provider = VaultMemoryProvider
