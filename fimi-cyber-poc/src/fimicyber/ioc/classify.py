"""IOC classification into 4 categories (spec 4.4)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

# ── Benign lists ─────────────────────────────────────────────────────────────

_BENIGN_DOMAINS: frozenset[str] = frozenset(
    {
        # Major platforms
        "google.com", "google.co.uk", "google.de", "google.fr",
        "youtube.com", "facebook.com", "instagram.com", "twitter.com",
        "x.com", "tiktok.com", "reddit.com", "linkedin.com",
        "t.me", "telegram.org", "vk.com", "ok.ru", "weibo.com",
        "pinterest.com", "tumblr.com", "discord.com", "discord.gg",
        "whatsapp.com", "signal.org", "wechat.com", "line.me",
        # Search engines
        "bing.com", "duckduckgo.com", "yahoo.com", "baidu.com", "yandex.ru",
        # Major news / media
        "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
        "nytimes.com", "washingtonpost.com", "theguardian.com",
        "lemonde.fr", "spiegel.de", "corriere.it", "elmundo.es",
        # Security / research orgs
        "mandiant.com", "qurium.org", "disinfo.eu", "eu.disinfo.eu",
        "graphika.com", "dfrlab.org", "euvsdisinfo.eu", "bellingcat.com",
        "viglinum.fr", "viginum.fr",
        # Government / inter-gov
        "europa.eu", "ec.europa.eu", "eeas.europa.eu",
        "nato.int", "un.org", "state.gov", "gov.uk",
        # Public DNS
        "8.8.8.8", "8.8.4.4", "1.1.1.1", "9.9.9.9",
        # URL shorteners (keep but low confidence)
        "bit.ly", "t.co", "ow.ly", "tinyurl.com", "buff.ly",
        # Academic / archive
        "web.archive.org", "archive.org", "doi.org", "jstor.org",
        "medium.com", "substack.com",
    }
)

# Social platform content URL prefixes
_SOCIAL_PREFIXES: tuple[str, ...] = (
    "https://twitter.com/",
    "https://x.com/",
    "https://www.facebook.com/",
    "https://facebook.com/",
    "https://www.instagram.com/",
    "https://instagram.com/",
    "https://www.youtube.com/",
    "https://youtube.com/",
    "https://www.tiktok.com/",
    "https://tiktok.com/",
    "https://t.me/",
    "https://vk.com/",
    "https://www.reddit.com/",
    "https://reddit.com/",
)

# RFC1918 private ranges
_PRIVATE_NETS = [
    re.compile(r"^10\."),
    re.compile(r"^172\.(1[6-9]|2\d|3[01])\."),
    re.compile(r"^192\.168\."),
    re.compile(r"^127\."),
    re.compile(r"^0\."),
    re.compile(r"^255\."),
]

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


def _load_extra_benign() -> frozenset[str]:
    path = Path(__file__).parent.parent.parent.parent / "data" / "curated" / "benign_extra.txt"
    if not path.exists():
        return frozenset()
    return frozenset(
        line.strip().lower()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    )


_EXTRA_BENIGN = _load_extra_benign()
_ALL_BENIGN = _BENIGN_DOMAINS | _EXTRA_BENIGN


def _is_private_ip(value: str) -> bool:
    return any(p.match(value) for p in _PRIVATE_NETS)


def _extract_domain(value: str) -> str:
    """Get bare domain from a URL or value."""
    value = re.sub(r"^https?://", "", value)
    return value.split("/")[0].split(":")[0].lower()


Category = Literal[
    "EvidenceSourceURL", "PlatformContentURL", "OperationalIOC", "BenignReference"
]
Status = Literal["candidate", "validated", "rejected", "needs_review"]


def classify_ioc(
    value: str,
    ioc_type: str,
    context: str,
    evidence_sources: list[str],
) -> tuple[Category, Status]:
    """Return (category, status) — priority order per spec 4.4."""

    domain = _extract_domain(value)

    # ── 1. Benign match (but evidence_sources take priority for URLs) ────
    # For URLs: check if it's an evidence source BEFORE benign-listing it.
    # A URL from a trusted org that is also listed as evidence_source should
    # be classified as EvidenceSourceURL, not BenignReference.
    if ioc_type in ("url", "tg_channel") and evidence_sources:
        if value in evidence_sources or any(
            value.startswith(src) or src.startswith(value)
            for src in evidence_sources
        ):
            return "EvidenceSourceURL", "rejected"

    if _is_benign(value, domain, ioc_type):
        return "BenignReference", "rejected"

    # Private IPs are benign
    if ioc_type == "ipv4" and _is_private_ip(value):
        return "BenignReference", "rejected"

    # ── 2. Social platform content URL ──────────────────────────────────
    if ioc_type in ("url", "tg_channel"):
        if any(value.startswith(prefix) for prefix in _SOCIAL_PREFIXES):
            return "PlatformContentURL", "rejected"

    # ── 3. Evidence source URL (non-benign-listed) ───────────────────────
    if ioc_type in ("url", "tg_channel"):
        if value in evidence_sources or any(
            value.startswith(src) or src.startswith(value)
            for src in evidence_sources
        ):
            return "EvidenceSourceURL", "rejected"

    # ── 4a. Operational keyword in context ──────────────────────────────
    op_count = len(_OP_KEYWORDS.findall(context))
    src_count = len(_SRC_KEYWORDS.findall(context))

    if src_count > op_count:
        return "EvidenceSourceURL", "rejected"

    if op_count > 0:
        return "OperationalIOC", "candidate"

    # ── 4b. Context ambiguous ────────────────────────────────────────────
    return "OperationalIOC", "needs_review"


def _is_benign(value: str, domain: str, ioc_type: str) -> bool:
    if domain in _ALL_BENIGN:
        return True
    if ioc_type == "ipv4" and value in _ALL_BENIGN:
        return True
    # Check TLD patterns for gov/europa
    if domain.endswith(".gov") or domain.endswith(".gov.uk"):
        return True
    if "europa.eu" in domain:
        return True
    return False
