"""
Markdown chunker — port of Nous vector-memory/chunker.ts

Splits markdown files into overlapping semantic chunks suitable for
embedding and BM25 indexing. Each chunk carries its nearest parent
heading for result attribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

TARGET_CHUNK_SIZE = 1500   # chars
OVERLAP_SIZE = 200         # chars — tail of previous chunk prepended to next
MIN_CHUNK_SIZE = 10        # chars — skip empty/junk fragments


@dataclass
class ChunkData:
    file_path: str
    file_mtime: int          # ms epoch — same as JS Date.now()
    chunk_index: int
    content: str
    heading: Optional[str]
    token_count: int         # rough estimate: len/4


def chunk_markdown(file_path: str, content: str, mtime_ms: int) -> list[ChunkData]:
    """Chunk a markdown file into overlapping text segments.

    Args:
        file_path:  Stored verbatim on each ChunkData.
        content:    Raw UTF-8 file content.
        mtime_ms:   File mtime as milliseconds epoch.

    Returns:
        List of ChunkData. Empty for blank or frontmatter-only files.
    """
    body = _strip_frontmatter(content)
    if not body.strip():
        return []

    sections = _split_into_sections(body)
    chunks: list[ChunkData] = []

    for section in sections:
        for text in _chunk_section(section["content"]):
            chunks.append(ChunkData(
                file_path=file_path,
                file_mtime=mtime_ms,
                chunk_index=len(chunks),
                content=text,
                heading=section["heading"],
                token_count=max(1, len(text) // 4),
            ))

    return chunks


# ── Internal helpers ──────────────────────────────────────────────────────────

def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter delimited by opening and closing ---."""
    if not content.startswith("---"):
        return content
    end = content.find("\n---", 3)
    if end == -1:
        return content  # malformed — treat as no frontmatter
    after = content[end + 4:]
    return after[1:] if after.startswith("\n") else after


def _split_into_sections(body: str) -> list[dict]:
    """Split body into sections at ## and ### headings.

    Content before the first heading is an anonymous section.
    Returns list of {"heading": str|None, "content": str}.
    """
    sections = []
    current_heading: Optional[str] = None
    buffer: list[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            sections.append({"heading": current_heading, "content": text})
        buffer.clear()

    for line in body.splitlines():
        m = re.match(r"^(#{2,3})\s+(.+)$", line)
        if m:
            flush()
            current_heading = m.group(2).strip()
        else:
            buffer.append(line)

    flush()
    return sections


def _chunk_section(content: str) -> list[str]:
    """Break a section into TARGET_CHUNK_SIZE chunks with OVERLAP_SIZE overlap.

    Prefers paragraph boundaries; force-splits long paragraphs at char level.
    Drops chunks below MIN_CHUNK_SIZE.
    """
    raw_paragraphs = re.split(r"\n\n+", content)

    # Force-split paragraphs that exceed TARGET_CHUNK_SIZE
    paragraphs: list[str] = []
    for para in raw_paragraphs:
        trimmed = para.strip()
        if not trimmed:
            continue
        if len(trimmed) <= TARGET_CHUNK_SIZE:
            paragraphs.append(trimmed)
        else:
            pos = 0
            while pos < len(trimmed):
                paragraphs.append(trimmed[pos:pos + TARGET_CHUNK_SIZE])
                pos += TARGET_CHUNK_SIZE

    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    overlap = ""

    for para in paragraphs:
        if not current:
            current = (overlap + "\n\n" + para) if overlap else para
        elif len(current) + 2 + len(para) <= TARGET_CHUNK_SIZE:
            current = current + "\n\n" + para
        else:
            if len(current) >= MIN_CHUNK_SIZE:
                chunks.append(current)
            overlap = current[-OVERLAP_SIZE:]
            current = overlap + "\n\n" + para

    if len(current) >= MIN_CHUNK_SIZE:
        chunks.append(current)

    return chunks
