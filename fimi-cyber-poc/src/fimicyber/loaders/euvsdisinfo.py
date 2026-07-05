"""EUvsDisinfo metadata loader.

The public Zenodo release contains article metadata and URL/topic fields, not
full article text. We map each article to the common Event schema and use the
EUvsDisinfo debunk ID as the unified link group for retrieval evaluation.
"""
from __future__ import annotations

import csv
import random
import re
import urllib.request
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Any

from fimicyber.schema import Event


DOWNLOAD_URL = "https://zenodo.org/records/10514307/files/euvsdisinfo_base.csv?download=1"


def load_events(
    raw_dir: Path,
    cfg: Any,
    fallbacks_used: list[str] | None = None,
) -> list[Event]:
    """Return EUvsDisinfo events from the Zenodo metadata CSV."""
    if fallbacks_used is None:
        fallbacks_used = []

    path = raw_dir / "euvsdisinfo" / "euvsdisinfo_base.csv"
    if not path.exists():
        try:
            _download(path)
        except Exception as exc:
            fallbacks_used.append(f"EUvsDisinfo download failed: {exc}")
            return []

    rows = _read_rows(path)
    if not rows:
        fallbacks_used.append("EUvsDisinfo CSV empty or unreadable")
        return []

    rows = _select_rows(rows, cfg)
    events = [_map_row(row, idx) for idx, row in enumerate(rows)]
    events = [ev for ev in events if ev is not None]
    return events


def _download(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as response:
        path.write_bytes(response.read())


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _select_rows(rows: list[dict[str, str]], cfg: Any) -> list[dict[str, str]]:
    ds_cfg = getattr(cfg, "datasets", {}).get("euvsdisinfo", {})
    max_events = ds_cfg.get("max_events", 450)
    min_group_size = int(ds_cfg.get("min_group_size", 2))

    if max_events in (None, "all", 0):
        return rows
    max_events = int(max_events)
    if max_events <= 0 or len(rows) <= max_events:
        return rows

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    singletons: list[dict[str, str]] = []
    for row in rows:
        debunk_id = (row.get("debunk_id") or "").strip()
        if debunk_id:
            grouped[debunk_id].append(row)
        else:
            singletons.append(row)

    group_ids = [
        gid for gid, items in grouped.items()
        if len(items) >= min_group_size
    ]
    rng = random.Random(int(getattr(cfg, "seed", 42)))
    rng.shuffle(group_ids)

    selected: list[dict[str, str]] = []
    for gid in group_ids:
        items = grouped[gid]
        if len(selected) + len(items) <= max_events:
            selected.extend(items)
        elif not selected and len(items) > max_events:
            selected.extend(items[:max_events])
            break
        if len(selected) >= max_events:
            break

    if len(selected) < max_events:
        used_ids = {id(row) for row in selected}
        rest = [
            row for row in rows
            if id(row) not in used_ids and row not in singletons
        ] + singletons
        selected.extend(rest[: max_events - len(selected)])

    return selected[:max_events]


def _map_row(row: dict[str, str], idx: int) -> Event | None:
    article_id = (row.get("article_id") or "").strip()
    debunk_id = (row.get("debunk_id") or "").strip()
    if not article_id or not debunk_id:
        return None

    keywords = _clean(row.get("keywords") or "unknown topic")
    publisher = _clean(row.get("article_publisher") or "")
    domain = _clean(row.get("article_domain") or "")
    url = (row.get("article_url") or "").strip()
    language = _clean(row.get("article_language") or "unknown")
    label = _clean(row.get("class") or "unknown")
    first_seen = _parse_date(row.get("debunk_date"))

    event_id = f"EUVS-{article_id}"
    title = f"EUvsDisinfo {label}: {keywords}"
    if domain:
        title = f"{title} ({domain})"

    # Keep retrieval text free of the ground-truth debunk_id.  The debunk_id is
    # used only as campaign_id below; putting it in description would leak the
    # answer label into the narrative embedding input.
    description_parts = [
        f"EUvsDisinfo metadata item labelled {label}.",
        f"Topics: {keywords}.",
        f"Language: {language}.",
    ]
    description = " ".join(part for part in description_parts if part)

    channels = ["web"]
    if language:
        channels.append(f"lang:{_norm_token(language)}")
    if domain:
        channels.append(f"domain:{domain.lower()}")

    target_sectors = [
        _norm_token(part)
        for part in re.split(r"[,;/|]", keywords)
        if _norm_token(part)
    ][:8]

    try:
        return Event(
            event_id=event_id,
            title=title,
            description=description,
            campaign_id=f"euvsdisinfo:debunk:{debunk_id}",
            campaign_id_source="debunk_group",
            reported_actor="pro-kremlin ecosystem" if label == "disinformation" else None,
            target_countries=[],
            target_sectors=target_sectors,
            first_seen=first_seen,
            last_seen=first_seen,
            ttps=[],
            channels=list(dict.fromkeys(channels)),
            evidence_sources=[url] if url else [],
            source_dataset="euvsdisinfo",
        )
    except Exception:
        return None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _norm_token(value: str) -> str:
    return re.sub(r"[^a-z0-9가-힣_]+", "_", value.strip().lower()).strip("_")
