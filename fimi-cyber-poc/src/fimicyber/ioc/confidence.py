"""IOCConfidence scorer (spec 4.5).

IOCConfidence(o) = 0.30·C_context + 0.25·C_source + 0.20·C_corroboration
                 + 0.15·C_type + 0.10·C_freshness
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Literal

# Keyword patterns (reused from classify.py to avoid circular import)
_OP_KEYWORDS = re.compile(
    r"\b(indicator|ioc|phishing|malware|c2|command[\s\-]and[\s\-]control|"
    r"spoofed?|typosquat|clone[d]?|registered|hosted|redirect(?:s|ed)? to|"
    r"infrastructure|fake\s+(?:site|domain|outlet|news)|impersonat\w*)\b",
    re.I,
)
_SRC_KEYWORDS = re.compile(
    r"\b(source|report|according to|reference|archive|published by)\b",
    re.I,
)

# Trusted source domains
_TRUSTED_ORGS: frozenset[str] = frozenset(
    {
        "qurium.org", "mandiant.com", "viginum.fr", "viglinum.fr",
        "euvsdisinfo.eu", "eu.disinfo.eu", "disinfo.eu",
        "graphika.com", "dfrlab.org", "bellingcat.com",
        "gstatic.com",  # Google TAG
        "cert.gov", "cisa.gov", "ncsc.gov.uk", "bsi.bund.de",
        "eeas.europa.eu", "europa.eu", "sec.gov",
        "meta.com", "googletag",
    }
)
_MAJOR_MEDIA: frozenset[str] = frozenset(
    {
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
        "nytimes.com", "washingtonpost.com", "theguardian.com",
        "lemonde.fr", "spiegel.de",
    }
)

# C_type lookup
_TYPE_WEIGHT: dict[str, float] = {
    "hash_sha256": 1.0, "hash_sha1": 1.0, "hash_md5": 1.0,
    "url": 1.0,
    "email": 0.9,
    "domain": 0.8,
    "ns": 0.6, "account": 0.6, "tg_channel": 0.6,
    "ipv4": 0.4,
    "asn": 0.3,
}


def compute_confidence(
    ioc_type: str,
    category: str,
    context: str,
    source_label: str,
    n_sources: int,
    event_first_seen: date | None,
    event_last_seen: date | None,
    ioc_first_seen: date | None,
    ioc_last_seen: date | None,
) -> tuple[dict[str, float], float]:
    """Return (components_dict, scalar_confidence)."""

    # C_context
    op_hits = len(_OP_KEYWORDS.findall(context))
    src_hits = len(_SRC_KEYWORDS.findall(context))
    if src_hits > op_hits:
        c_context = 0.1
    elif op_hits >= 2:
        c_context = 1.0
    elif op_hits == 1:
        c_context = 0.7
    else:
        c_context = 0.3

    # C_source
    c_source = _score_source(source_label)

    # C_corroboration
    if n_sources >= 3:
        c_corr = 1.0
    elif n_sources == 2:
        c_corr = 0.7
    else:
        c_corr = 0.3

    # C_type
    c_type = _TYPE_WEIGHT.get(ioc_type, 0.4)

    # C_freshness
    c_fresh = _score_freshness(
        event_first_seen, event_last_seen, ioc_first_seen, ioc_last_seen
    )

    components = {
        "C_context": round(c_context, 4),
        "C_source": round(c_source, 4),
        "C_corroboration": round(c_corr, 4),
        "C_type": round(c_type, 4),
        "C_freshness": round(c_fresh, 4),
    }
    confidence = round(
        0.30 * c_context
        + 0.25 * c_source
        + 0.20 * c_corr
        + 0.15 * c_type
        + 0.10 * c_fresh,
        6,
    )
    return components, confidence


def _score_source(source_label: str) -> float:
    label_lower = source_label.lower()
    if any(org in label_lower for org in _TRUSTED_ORGS):
        return 1.0
    if source_label == "curated_csv":
        return 1.0
    if any(m in label_lower for m in _MAJOR_MEDIA):
        return 0.7
    return 0.4


def _score_freshness(
    ev_first: date | None,
    ev_last: date | None,
    ioc_first: date | None,
    ioc_last: date | None,
) -> float:
    if ev_first is None or ioc_first is None:
        return 0.5  # unknown → neutral

    ev_end = ev_last or ev_first
    ioc_end = ioc_last or ioc_first

    # Overlap: [ev_first, ev_end] ∩ [ioc_first, ioc_end]
    overlap_start = max(ev_first, ioc_first)
    overlap_end = min(ev_end, ioc_end)
    if overlap_start <= overlap_end:
        return 1.0

    # Gap in days
    if ioc_first > ev_end:
        gap = (ioc_first - ev_end).days
    else:
        gap = (ev_first - ioc_end).days

    return round(math.exp(-gap / 180.0), 6)
