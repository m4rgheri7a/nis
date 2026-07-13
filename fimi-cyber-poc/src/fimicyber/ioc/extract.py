"""IOC extraction from event descriptions (spec 4.4).

Pipeline: refang → regex extraction → classify → confidence.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterator

import tldextract

from fimicyber.schema import IOC, Event

# ── defang/refang ────────────────────────────────────────────────────────────

_REFANG_PATTERNS = [
    (re.compile(r"hxxps?", re.I), lambda m: m.group(0).replace("xx", "tt").replace("XX", "TT")),
    (re.compile(r"\[dot\]", re.I), lambda m: "."),
    (re.compile(r"\(dot\)", re.I), lambda m: "."),
    (re.compile(r"\{dot\}", re.I), lambda m: "."),
    (re.compile(r"\[\.\]"), lambda m: "."),
    (re.compile(r"\(\.\)"), lambda m: "."),
    (re.compile(r"\{\.\}"), lambda m: "."),
    (re.compile(r"\[at\]", re.I), lambda m: "@"),
    (re.compile(r"\(@\)"), lambda m: "@"),
]


def refang(text: str) -> str:
    for pattern, repl in _REFANG_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def defang(value: str) -> str:
    """Re-defang an IOC value for display."""
    # Replace protocol first, then dots
    value = re.sub(r"^https(://)", r"hxxps\1", value, flags=re.I)
    value = re.sub(r"^http(://)", r"hxxp\1", value, flags=re.I)
    # Replace remaining dots (not in protocol part)
    value = re.sub(r"(?<![:/])\.(?!//)(?!//)", "[.]", value)
    return value


# ── regex patterns ────────────────────────────────────────────────────────────

_RE_IPV4 = re.compile(r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b")
_RE_URL = re.compile(r"https?://[^\s\"'<>)\]]+")
_RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_RE_HASH_MD5 = re.compile(r"\b[0-9a-fA-F]{32}\b")
_RE_HASH_SHA1 = re.compile(r"\b[0-9a-fA-F]{40}\b")
_RE_HASH_SHA256 = re.compile(r"\b[0-9a-fA-F]{64}\b")
_RE_TG_CHANNEL = re.compile(r"https?://t\.me/[A-Za-z0-9_@/]+")

# Operational context keywords (±120 chars window)
_OP_KEYWORDS = re.compile(
    r"\b(indicator|ioc|phishing|malware|c2|command[\s\-]and[\s\-]control|"
    r"spoofed|typosquat|clone[d]?|registered|hosted|redirect(?:s|ed)? to|"
    r"infrastructure|fake\s+(?:site|domain|outlet|news)|impersonat\w*)\b",
    re.I,
)
_SRC_KEYWORDS = re.compile(
    r"\b(source|report|according to|reference|archive|published by)\b",
    re.I,
)


def _validate_ipv4(m: re.Match) -> bool:
    return all(0 <= int(g) <= 255 for g in m.groups())


def _context(text: str, start: int, end: int, window: int = 120) -> str:
    return text[max(0, start - window) : end + window]


def _extract_raw_iocs(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield (value, ioc_type, context_snippet)."""
    clean = refang(text)

    # Remove URLs before domain extraction so they don't double-match
    url_spans: list[tuple[int, int]] = []

    # TG channels first (subset of URLs)
    for m in _RE_TG_CHANNEL.finditer(clean):
        yield m.group(0), "tg_channel", _context(clean, m.start(), m.end())
        url_spans.append((m.start(), m.end()))

    # URLs
    for m in _RE_URL.finditer(clean):
        if not any(s <= m.start() < e for s, e in url_spans):
            yield m.group(0), "url", _context(clean, m.start(), m.end())
            url_spans.append((m.start(), m.end()))

    # IPv4
    for m in _RE_IPV4.finditer(clean):
        if _validate_ipv4(m):
            yield m.group(0), "ipv4", _context(clean, m.start(), m.end())

    # Email
    for m in _RE_EMAIL.finditer(clean):
        yield m.group(0), "email", _context(clean, m.start(), m.end())

    # Hashes (longest first to avoid sha256 matching sha1 prefix)
    hash_spans: list[tuple[int, int]] = []
    for m in _RE_HASH_SHA256.finditer(clean):
        yield m.group(0), "hash_sha256", _context(clean, m.start(), m.end())
        hash_spans.append((m.start(), m.end()))
    for m in _RE_HASH_SHA1.finditer(clean):
        if not any(s <= m.start() < e for s, e in hash_spans):
            yield m.group(0), "hash_sha1", _context(clean, m.start(), m.end())
            hash_spans.append((m.start(), m.end()))
    for m in _RE_HASH_MD5.finditer(clean):
        if not any(s <= m.start() < e for s, e in hash_spans):
            yield m.group(0), "hash_md5", _context(clean, m.start(), m.end())

    # Domains (tokens not already matched as URL/IP/email/hash)
    covered = set()
    for s, e in url_spans:
        covered.update(range(s, e))
    for token_m in re.finditer(r"[^\s\"'<>()\[\]{},;]+", clean):
        if any(i in covered for i in range(token_m.start(), token_m.end())):
            continue
        token = token_m.group(0)
        try:
            ext = tldextract.extract(token)
            if ext.domain and ext.suffix and not token.startswith("@"):
                yield token, "domain", _context(clean, token_m.start(), token_m.end())
        except Exception:
            continue


def extract_iocs_from_event(event: Event) -> list[IOC]:
    """Extract IOCs from event.description. Returns OperationalIOC candidates."""
    return extract_iocs_from_text(event.description, event)


def extract_iocs_from_text(
    text: str,
    event: Event,
    source_label: str = "text_extraction",
) -> list[IOC]:
    """Extract IOCs from arbitrary case text using ``event`` only for metadata.

    Used by the evidence-structuring conditions, which read a reconstructed case
    dossier rather than the curated ``description`` field.
    """
    from fimicyber.ioc.classify import classify_ioc
    from fimicyber.ioc.confidence import compute_confidence

    if not text or not text.strip():
        return []

    seen: set[str] = set()
    result: list[IOC] = []

    for value, ioc_type, ctx in _extract_raw_iocs(text):
        key = f"{ioc_type}:{value}"
        if key in seen:
            continue
        seen.add(key)

        category, status = classify_ioc(
            value=value,
            ioc_type=ioc_type,
            context=ctx,
            evidence_sources=event.evidence_sources,
        )

        conf_components, confidence = compute_confidence(
            ioc_type=ioc_type,
            category=category,
            context=ctx,
            source_label=source_label,
            n_sources=1,
            event_first_seen=event.first_seen,
            event_last_seen=event.last_seen,
            ioc_first_seen=None,
            ioc_last_seen=None,
        )

        ioc = IOC(
            value=value,
            ioc_type=ioc_type,
            category=category,
            confidence=confidence,
            conf_components=conf_components,
            sources=event.evidence_sources[:3],
            status=status,
            synthetic=False,
        )
        result.append(ioc)

    return result
