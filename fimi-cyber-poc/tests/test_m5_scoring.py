"""M5 tests: T8, T9."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── T8: Jaccard / Overlap / mix ───────────────────────────────────────────────

def test_T8_jaccard():
    from fimicyber.scoring.components import jaccard
    a = {"t1", "t2"}
    b = {"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"}
    result = jaccard(a, b)
    assert abs(result - 0.2) < 1e-9, f"Expected 0.2, got {result}"


def test_T8_overlap():
    from fimicyber.scoring.components import overlap
    a = {"t1", "t2"}
    b = {"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"}
    result = overlap(a, b)
    assert abs(result - 1.0) < 1e-9, f"Expected 1.0, got {result}"


def test_T8_mix():
    from fimicyber.scoring.components import sim_sets
    a = {"t1", "t2"}
    b = {"t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10"}
    result = sim_sets(a, b, "mix")
    # 0.5*0.2 + 0.5*1.0 = 0.6
    assert abs(result - 0.6) < 1e-9, f"Expected 0.6, got {result}"


# ── T9: FCLS renormalisation golden value ─────────────────────────────────────

def test_T9_fcls_golden():
    """N=0.8, I=0.5, D=missing, C=0.2, T=1.0, ζ=0 → 0.6375."""
    from fimicyber.scoring.fcls import fcls

    components = {"N": 0.8, "I": 0.5, "D": None, "C": 0.2, "T": 1.0, "A": None}
    weights = {"N": 0.30, "I": 0.30, "D": 0.15, "C": 0.10, "T": 0.10, "A": 0.0}

    result = fcls(components, weights)
    # num = 0.30*0.8 + 0.30*0.5 + 0.10*0.2 + 0.10*1.0 = 0.24+0.15+0.02+0.10 = 0.51
    # den = 0.30+0.30+0.10+0.10 = 0.80
    # FCLS = 0.51/0.80 = 0.6375
    assert abs(result - 0.6375) < 1e-9, f"Expected 0.6375, got {result}"


def test_T9_fcls_all_missing():
    from fimicyber.scoring.fcls import fcls
    import math
    result = fcls({"N": None, "I": None}, {"N": 0.5, "I": 0.5})
    assert math.isnan(result)


def test_T9_fcls_no_zero_substitution():
    """Missing components must NOT be treated as 0."""
    from fimicyber.scoring.fcls import fcls

    # If D=missing were treated as 0:
    # FCLS = (0.30*0.8 + 0.30*0.5 + 0.15*0 + 0.10*0.2 + 0.10*1.0) / 1.0 = 0.51
    # With renorm (D excluded): 0.51/0.80 = 0.6375
    # These differ, so we can verify renorm is happening
    comp_with_zero = {"N": 0.8, "I": 0.5, "D": 0.0, "C": 0.2, "T": 1.0, "A": None}
    comp_missing   = {"N": 0.8, "I": 0.5, "D": None, "C": 0.2, "T": 1.0, "A": None}
    weights = {"N": 0.30, "I": 0.30, "D": 0.15, "C": 0.10, "T": 0.10, "A": 0.0}

    score_zero = fcls(comp_with_zero, weights)
    score_missing = fcls(comp_missing, weights)
    assert score_zero != score_missing, "Missing and zero should produce different scores"
