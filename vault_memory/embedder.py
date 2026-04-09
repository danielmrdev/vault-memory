"""
GeminiEmbedder — HTTP client for the Gemini text-embedding-004 API.

Port of Nous vector-memory/embedder.ts with identical behaviour:
  - Batches up to BATCH_SIZE texts per request
  - Inter-batch delay between consecutive calls (rate limit avoidance)
  - Exponential backoff on 429/5xx: 3 retries at 5s / 15s / 30s
  - is_available flag — False when key is empty or in rate-limit cooldown
  - Auto-recovery after RATE_LIMIT_COOLDOWN_S seconds
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

GEMINI_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-embedding-001:batchEmbedContents"
)

BATCH_SIZE = 20
INTER_BATCH_DELAY_S = 0.5
EMBEDDING_DIMS = 768
MAX_RETRIES = 3
RETRY_DELAYS_S = [5, 15, 30]
RATE_LIMIT_COOLDOWN_S = 10 * 60  # 10 min


class GeminiEmbedder:
    """Embed texts via Gemini API. Stateless except for rate-limit cooldown."""

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._cooldown_until: float = 0.0

    @property
    def is_available(self) -> bool:
        if not self._api_key:
            return False
        if time.time() < self._cooldown_until:
            remaining = int(self._cooldown_until - time.time())
            logger.debug(f"[Embedder] In cooldown for {remaining}s more")
            return False
        return True

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts. Returns list of 768-dim float vectors.

        Batches internally. Returns [] on total failure.
        On rate limit: enters cooldown, returns whatever was embedded so far.
        """
        if not self.is_available:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            chunk_indices = list(range(i, i + len(batch)))

            result = self._embed_with_retry(batch, chunk_indices)
            if result == "rate_limited":
                logger.warning("[Embedder] Rate limit — entering cooldown")
                self._cooldown_until = time.time() + RATE_LIMIT_COOLDOWN_S
                break
            elif result is None:
                logger.error(f"[Embedder] Batch {i}-{i+len(batch)} failed permanently")
                break
            else:
                all_embeddings.extend(result)

            if i + BATCH_SIZE < len(texts):
                time.sleep(INTER_BATCH_DELAY_S)

        return all_embeddings

    def _embed_with_retry(self, texts: list[str],
                          chunk_indices: list[int]) -> Optional[list[list[float]] | str]:
        """Try to embed texts with exponential backoff.

        Returns:
          list[list[float]]  — success
          "rate_limited"     — 429 after all retries
          None               — non-rate-limit failure after all retries
        """
        last_status = 0

        for attempt in range(MAX_RETRIES):
            result = self._call_api(texts)
            if result["ok"]:
                return result["embeddings"]

            last_status = result["status"]

            if attempt < MAX_RETRIES - 1 and (last_status == 429 or last_status >= 500):
                delay = RETRY_DELAYS_S[attempt]
                logger.warning(
                    f"[Embedder] Attempt {attempt+1}/{MAX_RETRIES} failed "
                    f"(HTTP {last_status}) for chunks {chunk_indices}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)
            else:
                break

        logger.error(
            f"[Embedder] All {MAX_RETRIES} retries exhausted for chunks "
            f"{chunk_indices}. Last status: HTTP {last_status}"
        )
        return "rate_limited" if last_status == 429 else None

    def _call_api(self, texts: list[str]) -> dict:
        """POST to Gemini batchEmbedContents. Returns {"ok": bool, ...}."""
        body = {
            "requests": [
                {
                    "model": "models/gemini-embedding-001",
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": EMBEDDING_DIMS,
                }
                for t in texts
            ]
        }
        data = json.dumps(body).encode()
        url = f"{GEMINI_EMBED_URL}?key={self._api_key}"
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return {"ok": False, "status": e.code}
        except Exception as e:
            logger.warning(f"[Embedder] Network error: {e}")
            return {"ok": False, "status": 503}

        embeddings = []
        for emb in payload.get("embeddings", []):
            values = emb.get("values", [])
            vec = values[:EMBEDDING_DIMS]
            # Pad to EMBEDDING_DIMS if short
            vec += [0.0] * (EMBEDDING_DIMS - len(vec))
            embeddings.append(vec)

        return {"ok": True, "embeddings": embeddings}
