"""
NvidiaEmbedder — HTTP client for the NVIDIA NIM embeddings API.

Model: nvidia/nv-embedqa-e5-v5
  - 1024 dims
  - Asymmetric (requires input_type: "passage" for docs, "query" for search)
  - Free tier via integrate.api.nvidia.com
  - ~340ms per batch of 20 on benchmarks

Behaviour (matching original GeminiEmbedder contract):
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

NVIDIA_EMBED_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"

BATCH_SIZE = 20
INTER_BATCH_DELAY_S = 0.2
EMBEDDING_DIMS = 1024
MAX_RETRIES = 3
RETRY_DELAYS_S = [5, 15, 30]
RATE_LIMIT_COOLDOWN_S = 60  # 1 min (NVIDIA free tier resets fast)


class NvidiaEmbedder:
    """Embed texts via NVIDIA NIM API. Stateless except for rate-limit cooldown."""

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

    def embed_batch(self, texts: list[str],
                    input_type: str = "passage") -> list[list[float]]:
        """Embed a list of texts. Returns list of 1024-dim float vectors.

        Args:
            texts: list of strings to embed
            input_type: "passage" for documents/chunks, "query" for search queries

        Batches internally (BATCH_SIZE=20).
        Returns [] on total failure.
        On rate limit: enters cooldown, returns whatever was embedded so far.
        """
        if not self.is_available:
            return []

        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            # Sanitize: encode UTF-8 then decode as ASCII with replacement
            # This removes emojis, box-drawing, etc that cause HTTP 400
            batch = [
                t.encode('utf-8').decode('ascii', errors='ignore')
                for t in batch
            ]
            chunk_indices = list(range(i, i + len(batch)))

            result = self._embed_with_retry(batch, chunk_indices, input_type)
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

    def _embed_with_retry(self, texts: list[str], chunk_indices: list[int],
                          input_type: str) -> Optional[list[list[float]] | str]:
        """Try to embed texts with exponential backoff.

        Returns:
          list[list[float]]  — success
          "rate_limited"     — 429 after all retries
          None               — non-rate-limit failure after all retries
        """
        last_status = 0

        for attempt in range(MAX_RETRIES):
            result = self._call_api(texts, input_type)
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

    def _call_api(self, texts: list[str], input_type: str) -> dict:
        """POST to NVIDIA NIM embeddings. Returns {"ok": bool, ...}."""
        body = {
            "input": texts,
            "model": NVIDIA_EMBED_MODEL,
            "input_type": input_type,
            "encoding_format": "float",
        }
        # Ensure UTF-8 encoding of JSON payload
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            NVIDIA_EMBED_URL, data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
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
        for item in payload.get("data", []):
            vec = item.get("embedding", [])
            vec = vec[:EMBEDDING_DIMS]
            vec += [0.0] * (EMBEDDING_DIMS - len(vec))
            embeddings.append(vec)

        return {"ok": True, "embeddings": embeddings}


# Backwards-compat alias
GeminiEmbedder = NvidiaEmbedder
