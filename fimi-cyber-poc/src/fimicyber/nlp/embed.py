"""Embedding store with SBERT / TF-IDF fallback (spec 5).

Caches embeddings in data/processed/embeddings.parquet keyed by text sha256.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import numpy as np

from fimicyber.schema import Event

_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")


def _clean_text(text: str) -> str:
    """Remove URLs from description for narrative purity."""
    return _URL_RE.sub(" ", text).strip()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbStore:
    """Manages embedding cache and encoding."""

    def __init__(self, cfg: Any, fallbacks_used: list[str] | None = None) -> None:
        self._cfg = cfg
        self._fallbacks_used = fallbacks_used or []
        self._cache_path = cfg.data_dir / "processed" / "embeddings.parquet"
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)

        self._backend = cfg.embedding.get("backend", "sbert")
        self._model_name = cfg.embedding.get("model", "paraphrase-multilingual-mpnet-base-v2")
        self._chunk_tokens = int(cfg.embedding.get("chunk_tokens", 220))
        self._chunk_overlap = int(cfg.embedding.get("chunk_overlap", 40))

        self._vectorizer: Any = None   # TF-IDF model if used
        self._sbert: Any = None        # SentenceTransformer if used

        # Load cache
        self._cache: dict[str, list[np.ndarray]] = {}  # sha256 → list of chunk vecs
        self._load_cache()

        self._new_encodings = 0  # count this run

    def encode_events(self, events: list[Event]) -> None:
        """Encode all events; skip those already cached."""
        to_encode = []
        for ev in events:
            clean = _clean_text(ev.description or "")
            key = _text_sha256(clean)
            if key not in self._cache:
                to_encode.append((key, clean, ev.event_id))

        if not to_encode:
            return

        self._ensure_backend()

        # For TF-IDF: fit on all texts before encoding any
        if self._backend == "tfidf":
            all_texts = [text for _, text, _ in to_encode]
            self._ensure_tfidf_fitted(all_texts)

        for key, text, eid in to_encode:
            chunks = self._chunk(text, eid)
            if not chunks:
                self._cache[key] = []
                continue
            vecs = self._encode_chunks(chunks)
            self._cache[key] = vecs
            self._new_encodings += 1

        self._save_cache()

    def get_vecs(self, event: Event) -> list[np.ndarray] | None:
        """Return chunk vectors for event, or None if missing/empty."""
        clean = _clean_text(event.description or "")
        if not clean:
            return None
        key = _text_sha256(clean)
        return self._cache.get(key)

    @property
    def new_encodings(self) -> int:
        return self._new_encodings

    # ── chunking ──────────────────────────────────────────────────────────

    def _chunk(self, text: str, eid: str) -> list[str]:
        """Split text into overlapping token-level chunks."""
        if not text.strip():
            return []

        if self._backend == "sbert" and self._sbert is not None:
            tokenizer = self._sbert.tokenizer
            tokens = tokenizer.encode(text, add_special_tokens=False)
            step = self._chunk_tokens - self._chunk_overlap
            if len(tokens) <= self._chunk_tokens:
                return [text]
            chunks: list[str] = []
            for start in range(0, len(tokens), step):
                chunk_toks = tokens[start : start + self._chunk_tokens]
                chunk_text = tokenizer.decode(chunk_toks, skip_special_tokens=True)
                if chunk_text.strip():
                    chunks.append(chunk_text)
            return chunks or [text]
        else:
            # TF-IDF fallback: paragraph splitting
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            return paragraphs if paragraphs else [text]

    def _encode_chunks(self, chunks: list[str]) -> list[np.ndarray]:
        if not chunks:
            return []
        if self._backend == "sbert" and self._sbert is not None:
            vecs = self._sbert.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            return [vecs[i] for i in range(len(vecs))]
        else:
            # TF-IDF: use the fitted vectorizer
            X = self._vectorizer.transform(chunks)
            return [np.asarray(X[i].todense()).flatten() for i in range(X.shape[0])]

    # ── backend init ──────────────────────────────────────────────────────

    def _ensure_backend(self) -> None:
        if self._backend == "sbert":
            self._try_load_sbert()
        if self._backend == "tfidf" or self._sbert is None:
            self._init_tfidf()

    def _try_load_sbert(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert = SentenceTransformer(self._model_name)
        except Exception as exc:
            self._sbert = None
            self._backend = "tfidf"
            msg = f"SBERT unavailable ({exc}) → TF-IDF fallback"
            self._fallbacks_used.append(msg)

    def _init_tfidf(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        # Fit on all cached texts + new texts
        all_texts: list[str] = []
        for vecs in self._cache.values():
            # We don't have the original texts from cache; use placeholder
            pass

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(
                max_features=4096, sublinear_tf=True, min_df=1
            )
            # We need to fit on actual texts; store them for this purpose
            # The vectorizer will be fit lazily on first use
            self._tfidf_fitted = False
        self._backend = "tfidf"

    def _ensure_tfidf_fitted(self, texts: list[str]) -> None:
        if not getattr(self, "_tfidf_fitted", False):
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(
                max_features=4096, sublinear_tf=True, min_df=1
            )
            fit_texts = texts if texts else ["placeholder"]
            self._vectorizer.fit(fit_texts)
            self._tfidf_fitted = True

    # ── cache I/O ─────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            import pandas as pd
            df = pd.read_parquet(self._cache_path)
            for row in df.itertuples():
                key = row.sha256
                vec = np.array(row.vector, dtype=np.float32)
                chunk_id = row.chunk_id
                if key not in self._cache:
                    self._cache[key] = []
                # Ensure list is long enough
                while len(self._cache[key]) <= chunk_id:
                    self._cache[key].append(None)
                self._cache[key][chunk_id] = vec
            # Clean up Nones
            for key in self._cache:
                self._cache[key] = [v for v in self._cache[key] if v is not None]
        except Exception:
            self._cache = {}

    def _save_cache(self) -> None:
        import pandas as pd

        rows = []
        for sha, vecs in self._cache.items():
            for chunk_id, vec in enumerate(vecs):
                rows.append({"sha256": sha, "chunk_id": chunk_id, "vector": vec.tolist()})

        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(self._cache_path, index=False)


# ── TF-IDF batch encode helper ─────────────────────────────────────────────

def tfidf_encode_all(events: list[Event]) -> EmbStore:
    """Return an EmbStore with TF-IDF embeddings fit on all event descriptions."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [_clean_text(ev.description or "") for ev in events]
    vectorizer = TfidfVectorizer(max_features=4096, sublinear_tf=True, min_df=1)
    X = vectorizer.fit_transform(texts)

    class _TfIdfStore:
        """Minimal EmbStore interface for TF-IDF."""
        def __init__(self):
            self._vecs = {}
            self.new_encodings = len(texts)

        def get_vecs(self, event: Event) -> list[np.ndarray] | None:
            clean = _clean_text(event.description or "")
            if not clean:
                return None
            return self._vecs.get(event.event_id)

        def encode_events(self, evs: list[Event]) -> None:
            pass

    store = _TfIdfStore()
    for i, ev in enumerate(events):
        vec = np.asarray(X[i].todense()).flatten().astype(np.float32)
        store._vecs[ev.event_id] = [vec]
    return store  # type: ignore
