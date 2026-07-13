"""Load source-separated external case data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fimicyber.loaders.curated import _attach_iocs, _event_from_json
from fimicyber.schema import Event


def load_external_case(cfg: Any) -> list[Event]:
    case_cfg = cfg.attribution.get("external_case", {})
    if case_cfg.get("enabled", True) is False:
        return []
    case_dir = cfg.data_dir / "external"
    events_path = case_dir / case_cfg.get("events_file", "ghostwriter_events.jsonl")
    iocs_path = case_dir / case_cfg.get("iocs_file", "ghostwriter_iocs.csv")
    if not events_path.exists():
        return []
    events = [
        _event_from_json(json.loads(line))
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if iocs_path.exists():
        _attach_iocs(events, iocs_path)
    return events


def load_multiactor_benchmark(cfg: Any) -> list[Event]:
    """Load the frozen four-class external generalisation benchmark."""
    benchmark_cfg = cfg.attribution.get("generalization_benchmark", {})
    if benchmark_cfg.get("enabled", True) is False:
        return []

    case_dir = cfg.data_dir / "external"
    event_files = benchmark_cfg.get(
        "events_files",
        ["ghostwriter_events.jsonl", "multiactor_events.jsonl"],
    )
    ioc_files = benchmark_cfg.get(
        "iocs_files",
        ["ghostwriter_iocs.csv", "multiactor_iocs.csv"],
    )

    events: list[Event] = []
    seen_ids: set[str] = set()
    for filename in event_files:
        path = case_dir / filename
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            event = _event_from_json(json.loads(line))
            if event.event_id in seen_ids:
                raise ValueError(f"Duplicate benchmark event_id: {event.event_id}")
            seen_ids.add(event.event_id)
            events.append(event)

    for filename in ioc_files:
        path = case_dir / filename
        if path.exists():
            _attach_iocs(events, path)
    return events
