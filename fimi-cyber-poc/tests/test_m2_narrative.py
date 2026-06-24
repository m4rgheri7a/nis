"""M2 tests: T5."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── T5: N combination formula ─────────────────────────────────────────────────

def test_T5_narrative_combination():
    """T5: max=0.9, avg_top3=0.6, λ=0.6 → 0.78 (spec golden value)."""
    from fimicyber.nlp.narrative import _pair_score

    # Use 1 chunk for event_i and 3 chunks for event_j.
    # This gives exactly 3 cross-pairs, making max and avg_top3 deterministic.
    # Target normalised cosines: 0.9, 0.6, 0.3
    # → raw cosines:            0.8, 0.2, -0.4

    def _make_unit(raw_cos: float) -> np.ndarray:
        """Return unit vector b such that [1,0]·b = raw_cos."""
        sin = (max(0.0, 1.0 - raw_cos**2)) ** 0.5
        return np.array([raw_cos, sin])

    # event_i: 1 chunk = unit vector along x
    vec_i = np.array([1.0, 0.0])

    # event_j: 3 chunks with decreasing cosine similarity
    vec_j0 = _make_unit(0.8)   # normalised cos = (0.8+1)/2 = 0.9
    vec_j1 = _make_unit(0.2)   # normalised cos = (0.2+1)/2 = 0.6
    vec_j2 = _make_unit(-0.4)  # normalised cos = (-0.4+1)/2 = 0.3

    vecs_i = [vec_i]
    vecs_j = [vec_j0, vec_j1, vec_j2]

    score = _pair_score(vecs_i, vecs_j, lam=0.6, top_k=3)
    expected = 0.6 * 0.9 + 0.4 * 0.6  # = 0.54 + 0.24 = 0.78
    assert abs(score - expected) < 1e-4, f"Expected {expected}, got {score}"


def test_T5_narrative_matrix_symmetry_and_diagonal(cfg):
    """N matrix must be symmetric and have NaN on diagonal."""
    from fimicyber.schema import Event
    from fimicyber.nlp.embed import EmbStore
    from fimicyber.nlp.narrative import narrative_matrix

    # Build minimal events with synthetic embeddings
    events = [
        Event(event_id="e1", title="t1", description="Phishing campaign targeting banks."),
        Event(event_id="e2", title="t2", description="Malware distributed via fake sites."),
        Event(event_id="e3", title="t3", description=""),  # missing → NaN
    ]

    # Monkey-patch EmbStore.get_vecs for test
    class _FakeEmb:
        new_encodings = 0
        def encode_events(self, evs): pass
        def get_vecs(self, ev):
            if ev.event_id == "e3":
                return None
            rng = __import__("numpy").random.RandomState(hash(ev.event_id) % (2**31))
            return [rng.randn(16).astype("float32")]

    mat = narrative_matrix(events, _FakeEmb(), cfg)

    n = len(events)
    assert mat.shape == (n, n)
    # Diagonal NaN
    for i in range(n):
        assert np.isnan(mat[i, i])
    # Symmetric
    for i in range(n):
        for j in range(n):
            if np.isnan(mat[i, j]):
                assert np.isnan(mat[j, i])
            else:
                assert abs(mat[i, j] - mat[j, i]) < 1e-7
    # Missing description → NaN row/col
    assert np.isnan(mat[2, 0])
    assert np.isnan(mat[0, 2])
