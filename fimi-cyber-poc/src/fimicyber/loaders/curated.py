"""Curated public-IOC case loader for the PoC.

Events are stored in JSONL and validated OperationalIOCs are stored in CSV with
public source references. The initial curated case uses Qurium/EU DisinfoLab's
Doppelganger infrastructure reporting so the PoC contains at least one real
IOC-backed evidence graph.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any

from fimicyber.ioc.confidence import compute_confidence
from fimicyber.schema import Event, IOC

DEFAULT_EVENTS_FILE = "curated_events.jsonl"
DEFAULT_IOCS_FILE = "real_iocs.csv"


def load_events(curated_dir: Path, cfg: Any, fallbacks_used: list[str] | None = None) -> list[Event]:
    """Load manually curated public IOC-backed events."""
    if fallbacks_used is None:
        fallbacks_used = []

    ds_cfg = getattr(cfg, "datasets", {}).get("curated", {})
    if ds_cfg.get("enabled", True) is False:
        return []

    events_path = curated_dir / ds_cfg.get("events_file", DEFAULT_EVENTS_FILE)
    iocs_path = curated_dir / ds_cfg.get("iocs_file", DEFAULT_IOCS_FILE)
    if not events_path.exists():
        fallbacks_used.append(f"curated events file missing: {events_path}")
        return []

    events: list[Event] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        events.append(_event_from_json(json.loads(line)))

    if iocs_path.exists():
        _attach_iocs(events, iocs_path)
    else:
        fallbacks_used.append(f"curated IOC file missing: {iocs_path}")
    return events


def _event_from_json(obj: dict[str, Any]) -> Event:
    return Event(
        event_id=str(obj["event_id"]),
        title=str(obj.get("title") or obj["event_id"]),
        description=str(obj.get("description") or obj.get("title") or obj["event_id"]),
        campaign_id=obj.get("campaign_id"),
        campaign_id_source=obj.get("campaign_id_source", "curated"),
        reported_actor=obj.get("reported_actor"),
        target_countries=list(obj.get("target_countries", [])),
        target_sectors=list(obj.get("target_sectors", [])),
        first_seen=_parse_date(obj.get("first_seen")),
        last_seen=_parse_date(obj.get("last_seen")),
        ttps=list(obj.get("ttps", [])),
        channels=list(obj.get("channels", [])),
        evidence_sources=list(obj.get("evidence_sources", [])),
        evidence_ids=list(obj.get("evidence_ids", [])),
        source_dataset=str(obj.get("source_dataset", "curated")),
        evaluation_role=obj.get("evaluation_role", "development"),
        date_basis=obj.get("date_basis", "observed"),
    )


def _attach_iocs(events: list[Event], iocs_path: Path) -> None:
    by_id = {ev.event_id: ev for ev in events}
    with iocs_path.open(encoding="utf-8", newline="") as f:
        rows = [dict(row) for row in csv.DictReader(f)]

    seen_by_event: dict[str, set[tuple[str, str]]] = {ev.event_id: set() for ev in events}
    for row in rows:
        event_id = (row.get("event_id") or "").strip()
        ev = by_id.get(event_id)
        if ev is None:
            continue

        value = (row.get("ioc_value") or "").strip()
        ioc_type = (row.get("ioc_type") or "domain").strip()
        category = (row.get("category") or "OperationalIOC").strip()
        if not value:
            continue

        key = (ioc_type, value)
        if key in seen_by_event[event_id]:
            continue
        seen_by_event[event_id].add(key)

        first_seen = _parse_date(row.get("first_seen"))
        last_seen = _parse_date(row.get("last_seen")) or first_seen
        source_url = (row.get("source_url") or "").strip()
        source_label = (row.get("source_label") or source_url or "curated_csv").strip()
        context = (row.get("context") or "Curated validated public IOC.").strip()
        confidence_raw = (row.get("confidence") or "").strip()
        if confidence_raw:
            conf_components = {"curated": 1.0}
            confidence = float(confidence_raw)
        else:
            unique_sources = {
                (r.get("source_url") or r.get("source_label") or "curated_csv").strip()
                for r in rows
                if (r.get("ioc_value") or "").strip() == value
            }
            conf_components, confidence = compute_confidence(
                ioc_type=ioc_type,
                category=category,
                context=context,
                source_label=source_label,
                n_sources=max(1, len(unique_sources)),
                event_first_seen=ev.first_seen,
                event_last_seen=ev.last_seen,
                ioc_first_seen=first_seen,
                ioc_last_seen=last_seen,
            )

        ev.iocs.append(IOC(
            value=value,
            ioc_type=ioc_type,
            category=category,
            confidence=max(0.0, min(1.0, confidence)),
            conf_components=conf_components,
            first_seen=first_seen,
            last_seen=last_seen,
            sources=[source_url] if source_url else [],
            evidence_ids=[row["evidence_id"].strip()] if row.get("evidence_id", "").strip() else [],
            status=row.get("status") or "validated",
            synthetic=False,
        ))


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        return None
