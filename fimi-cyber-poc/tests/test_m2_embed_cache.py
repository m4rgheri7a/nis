"""Embedding backend and cache-isolation tests."""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from fimicyber.nlp.embed import EmbStore


def _cfg(tmp_path):
    return SimpleNamespace(
        data_dir=tmp_path,
        embedding={
            "backend": "tfidf",
            "model": "test-model",
            "chunk_tokens": 220,
            "chunk_overlap": 40,
        },
    )


def test_empty_fallback_list_is_preserved(tmp_path):
    fallbacks = []
    store = EmbStore(_cfg(tmp_path), fallbacks)
    assert store._fallbacks_used is fallbacks
    assert store.backend == "tfidf"


def test_legacy_cache_without_backend_namespace_is_ignored(tmp_path):
    cache = tmp_path / "processed" / "embeddings.parquet"
    cache.parent.mkdir(parents=True)
    pd.DataFrame([{
        "sha256": "legacy",
        "chunk_id": 0,
        "vector": [1.0, 0.0],
    }]).to_parquet(cache, index=False)

    store = EmbStore(_cfg(tmp_path))
    assert store._cache == {}


def test_cache_namespace_includes_chunk_settings(tmp_path):
    store = EmbStore(_cfg(tmp_path))
    assert "chunk=220" in store._cache_namespace
    assert "overlap=40" in store._cache_namespace
