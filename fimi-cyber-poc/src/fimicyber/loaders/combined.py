"""Combined input loader for the FIMI-Cyber PoC."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from fimicyber.schema import Event
from fimicyber.loaders.disinfox import load_events as load_disinfox_events
from fimicyber.loaders.euvsdisinfo import load_events as load_euvsdisinfo_events
from fimicyber.loaders.curated import load_events as load_curated_events


def load_events(
    raw_dir: Path,
    cfg: Any,
    fallbacks_used: list[str] | None = None,
) -> list[Event]:
    """Load DISINFOX and EUvsDisinfo into one event pool."""
    if fallbacks_used is None:
        fallbacks_used = []

    disinfox_events = load_disinfox_events(raw_dir, cfg, fallbacks_used)
    euvs_events = load_euvsdisinfo_events(raw_dir, cfg, fallbacks_used)
    curated_events = load_curated_events(cfg.data_dir / "curated", cfg, fallbacks_used)

    events = _dedupe_event_ids(disinfox_events + euvs_events + curated_events)
    _write_mapping_report(cfg.data_dir / "interim", events)
    return events


def _dedupe_event_ids(events: list[Event]) -> list[Event]:
    seen: Counter[str] = Counter()
    result: list[Event] = []
    for ev in events:
        seen[ev.event_id] += 1
        if seen[ev.event_id] == 1:
            result.append(ev)
            continue
        new_id = f"{ev.event_id}-{seen[ev.event_id]}"
        result.append(ev.model_copy(update={"event_id": new_id}))
    return result


def _write_mapping_report(out_dir: Path, events: list[Event]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    source_counts = Counter(ev.source_dataset for ev in events)
    source_by_gt = Counter(ev.campaign_id_source for ev in events)
    campaigns = sorted({ev.campaign_id for ev in events if ev.campaign_id})
    eligible = [
        camp for camp in campaigns
        if sum(1 for ev in events if ev.campaign_id == camp) >= 2
    ]

    lines = [
        "# Combined Dataset → Event Schema Mapping Report",
        "",
        "**Input mode**: `DISINFOX + EUvsDisinfo + curated public IOC case` as one unified event pool",
        f"**Total events**: {len(events)}",
        f"**Events with link group**: {sum(1 for e in events if e.campaign_id)}",
        f"**Unique link groups**: {len(campaigns)}",
        f"**Evaluation-eligible link groups (size≥2)**: {len(eligible)}",
        "",
        "## Source Counts",
        "",
        "| source_dataset | events |",
        "|---|---:|",
    ]
    for source, count in sorted(source_counts.items()):
        lines.append(f"| {source} | {count} |")

    lines += [
        "",
        "## Link Group Sources",
        "",
        "| campaign_id_source | events |",
        "|---|---:|",
    ]
    for source, count in sorted(source_by_gt.items()):
        lines.append(f"| {source} | {count} |")

    lines += [
        "",
        "## Mapping Decisions",
        "",
        "| Dataset | Internal mapping |",
        "|---|---|",
        "| DISINFOX | `Threat Actor` is retained as `actor_surrogate` link group when no explicit campaign exists. |",
        "| EUvsDisinfo | `debunk_id` is mapped to `euvsdisinfo:debunk:<id>` for evaluation only; it is deliberately excluded from `description` to prevent label leakage. |",
        "| Curated Doppelganger | Public Qurium/EU DisinfoLab IOC values are attached as non-synthetic `OperationalIOC` evidence with `campaign_id_source=curated`. |",
        "| Combined evaluation | All events are ranked in one pool; no dataset-specific metrics are emitted. |",
        "| Synthetic IOC | Reserved synthetic IOCs are injected only into the original DISINFOX/fixture branch, not into EUvsDisinfo metadata rows. |",
        "",
    ]

    (out_dir / "mapping_report.md").write_text("\n".join(lines), encoding="utf-8")
