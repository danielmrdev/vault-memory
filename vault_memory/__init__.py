"""vault-memory — semantic search over collections of markdown files."""

from .indexer import VaultMemory, FileIndexResult, IndexSummary
from .store import SearchResult, VectorStore
from .chunker import ChunkData, chunk_markdown
from .embedder import GeminiEmbedder

__version__ = "0.1.0"
__all__ = [
    "VaultMemory", "FileIndexResult", "IndexSummary",
    "SearchResult", "VectorStore",
    "ChunkData", "chunk_markdown",
    "GeminiEmbedder",
]
