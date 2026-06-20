"""DISINFOX loader (spec 4.1).

Attempts to clone and parse the DISINFOX repository.
Falls back to fixture generator if data files cannot be found.
"""
from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

from fimicyber.schema import Event


def load_events(
    raw_dir: Path,
    cfg: Any,
    fallbacks_used: list[str] | None = None,
) -> list[Event]:
    """Return list[Event] from DISINFOX or fixture fallback."""
    if fallbacks_used is None:
        fallbacks_used = []

    disinfox_dir = raw_dir / "disinfox"

    # ── Step 1: clone if not already present ─────────────────────────────
    if not disinfox_dir.exists():
        ok = _clone(disinfox_dir)
        if not ok:
            fallbacks_used.append("DISINFOX clone failed → using fixtures")
            return _fixture_fallback(cfg.data_dir / "interim", fallbacks_used)

    # ── Step 2: find data files ───────────────────────────────────────────
    events = _try_parse(disinfox_dir)
    if events is None:
        fallbacks_used.append("DISINFOX data files not parseable → using fixtures")
        return _fixture_fallback(cfg.data_dir / "interim", fallbacks_used)

    if len(events) < 5:
        fallbacks_used.append(
            f"DISINFOX yielded only {len(events)} events → using fixtures"
        )
        return _fixture_fallback(cfg.data_dir / "interim", fallbacks_used)

    _write_mapping_report(cfg.data_dir / "interim", events, source="disinfox")
    return events


# ── Internal helpers ───────────────────────────────────────────────────────

def _clone(target: Path) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                "git", "clone", "--depth", "1",
                "https://github.com/CyberDataLab/disinfox",
                str(target),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0
    except Exception:
        return False


def _try_parse(disinfox_dir: Path) -> list[Event] | None:
    """Scan DISINFOX repo for incident data files and parse them."""
    # Try the known DSRM CSV first (151 rows, rich TTP data)
    dsrm_csv = disinfox_dir / "backend" / "data" / "merged_Foulde_DSRM_additions.csv"
    if dsrm_csv.exists():
        events = _parse_dsrm_csv(dsrm_csv)
        if events and len(events) >= 5:
            return events

    # Fallback: scan all JSON/CSV candidates
    candidates: list[Path] = []
    for pattern in ["**/*.json", "**/*.jsonl", "**/*.csv"]:
        candidates.extend(disinfox_dir.glob(pattern))

    for path in sorted(candidates):
        try:
            events = _parse_file(path)
            if events and len(events) >= 5:
                return events
        except Exception:
            continue
    return None


# ── DSRM CSV parser (primary DISINFOX data source) ──────────────────────────

_PLATFORM_COLS = [
    "Facebook", "Instagram", "X", "Youtube", "TikTok", "Telegram",
    "Gab", "Parler", "Gettr", "Truth Social", "Vkontakte",
    "Odnoklassniki", "Reddit", "4chan", "Discord", "Tumblr",
    "Pinterest", "WhatsApp", "WeChat", "Line",
]

# Map platform names to lowercase channel identifiers
_PLATFORM_MAP = {
    "X": "twitter",
    "Vkontakte": "vk",
    "Odnoklassniki": "ok",
    "Truth Social": "truth_social",
}


def _parse_dsrm_csv(path: Path) -> list[Event] | None:
    import csv, re

    events: list[Event] = []
    with open(path, encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # Identify TTP columns: start with "T" followed by 4+ digits
        ttp_cols = [h for h in headers if re.match(r"T\d{4}", h.strip())]

        for i, row in enumerate(reader):
            year = row.get("Year", "").strip()
            country = row.get("Target Country", "").strip()
            event_type = row.get("Event", "").strip()
            description = row.get("Event description", "").strip()
            actor = row.get("Threat Actor", "").strip()
            origin = row.get("Country of Origin", "").strip()

            if not description:
                description = f"{event_type} in {country} ({year})"

            title = f"{event_type} — {country} {year}"
            if actor and actor != "Unknown":
                title = f"[{actor}] {title}"

            event_id = f"DSRM-{i+1:04d}"

            # campaign_id: use Threat Actor as surrogate
            # (no explicit campaign field in this dataset)
            cleaned_actor = re.sub(r"[\*\s]+", "_", actor.strip().lower())
            campaign_id = f"actor:{cleaned_actor}" if actor and actor != "Unknown" else None

            # TTPs: columns with value "1"
            ttps = []
            for col in ttp_cols:
                if row.get(col, "").strip() == "1":
                    # Extract TTP ID (e.g., "T0022")
                    m = re.match(r"(T\d{4}(?:\.\d+)?)", col.strip())
                    if m:
                        ttps.append(m.group(1))

            # Channels
            channels = []
            for col in _PLATFORM_COLS:
                if row.get(col, "").strip() == "1":
                    channels.append(_PLATFORM_MAP.get(col, col.lower()))

            # Evidence sources
            evidence_sources = []
            for k in range(1, 10):
                src = row.get(f"Source {k}", "").strip()
                if src and src not in ("#NAME?", "0", ""):
                    if src.startswith("http"):
                        evidence_sources.append(src)

            # Date
            first_seen = _parse_date(year) if year else None
            last_seen = first_seen

            # Target country → ISO list
            target_countries = []
            if country:
                # Map common names to ISO codes where easy
                _COUNTRY_ISO = {
                    "Scotland": "GB", "United Kingdom": "GB", "UK": "GB",
                    "United States": "US", "USA": "US", "Brazil": "BR",
                    "France": "FR", "Germany": "DE", "Ukraine": "UA",
                    "Russia": "RU", "China": "CN", "Italy": "IT",
                    "Poland": "PL", "Lithuania": "LT", "Latvia": "LV",
                    "Estonia": "EE", "Sweden": "SE", "Norway": "NO",
                    "Finland": "FI", "Netherlands": "NL", "Belgium": "BE",
                    "Austria": "AT", "Spain": "ES", "Romania": "RO",
                    "Hungary": "HU", "Czech Republic": "CZ", "Slovakia": "SK",
                    "Bulgaria": "BG", "Greece": "GR", "Montenegro": "ME",
                    "North Macedonia": "MK", "Albania": "AL", "Kosovo": "XK",
                    "Serbia": "RS", "Bosnia": "BA", "Croatia": "HR",
                    "Slovenia": "SI", "Georgia": "GE", "Moldova": "MD",
                    "Belarus": "BY", "Kazakhstan": "KZ", "Turkey": "TR",
                    "Canada": "CA", "Australia": "AU", "New Zealand": "NZ",
                    "Japan": "JP", "South Korea": "KR", "India": "IN",
                    "Taiwan": "TW", "Israel": "IL", "Iran": "IR",
                    "EU": "EU", "Europe": "EU",
                }
                iso = _COUNTRY_ISO.get(country, country[:3].upper())
                target_countries = [iso]

            # Sector heuristic from event type
            target_sectors = []
            et_lower = event_type.lower()
            if "election" in et_lower or "referendum" in et_lower or "vote" in et_lower:
                target_sectors.append("elections")
            if "war" in et_lower or "military" in et_lower or "conflict" in et_lower:
                target_sectors.append("military")

            try:
                ev = Event(
                    event_id=event_id,
                    title=title,
                    description=description,
                    campaign_id=campaign_id,
                    reported_actor=actor or None,
                    target_countries=target_countries,
                    target_sectors=target_sectors,
                    first_seen=first_seen,
                    last_seen=last_seen,
                    ttps=list(dict.fromkeys(ttps)),  # deduplicate, preserve order
                    channels=channels,
                    evidence_sources=evidence_sources,
                    source_dataset="disinfox",
                )
                events.append(ev)
            except Exception:
                continue

    return events if events else None


def _parse_file(path: Path) -> list[Event] | None:
    suffix = path.suffix.lower()
    if suffix in (".json", ".jsonl"):
        return _parse_json(path)
    if suffix == ".csv":
        return _parse_csv(path)
    return None


def _parse_json(path: Path) -> list[Event] | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try JSONL
        events: list[Event] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ev = _map_record(obj)
                if ev:
                    events.append(ev)
            except Exception:
                continue
        return events if events else None

    if isinstance(data, list):
        events = []
        for obj in data:
            ev = _map_record(obj)
            if ev:
                events.append(ev)
        return events if events else None

    if isinstance(data, dict):
        # Could be {incidents: [...]} or similar
        for key in ("incidents", "events", "data", "results", "items"):
            if key in data and isinstance(data[key], list):
                events = []
                for obj in data[key]:
                    ev = _map_record(obj)
                    if ev:
                        events.append(ev)
                return events if events else None

    return None


def _parse_csv(path: Path) -> list[Event] | None:
    import csv

    events: list[Event] = []
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ev = _map_record(dict(row))
                if ev:
                    events.append(ev)
    except Exception:
        return None
    return events if events else None


def _map_record(obj: dict[str, Any]) -> Event | None:
    """Best-effort mapping from arbitrary DISINFOX record to Event schema."""
    if not isinstance(obj, dict):
        return None

    # event_id
    event_id = str(
        obj.get("id") or obj.get("_id") or obj.get("incident_id") or obj.get("event_id") or ""
    ).strip()
    if not event_id:
        return None

    # title & description
    title = str(obj.get("title") or obj.get("name") or obj.get("headline") or "").strip()
    description = str(
        obj.get("description") or obj.get("content") or obj.get("summary") or title
    ).strip()
    if not title and not description:
        return None
    if not title:
        title = description[:120]

    # campaign_id (ground truth)
    campaign_id = (
        str(obj.get("campaign_id") or obj.get("campaign") or "").strip() or None
    )
    if not campaign_id:
        # Try to infer from related-incidents structure
        related = obj.get("related_incidents") or obj.get("related") or []
        if isinstance(related, list) and related:
            # Use reported_actor as surrogate if no explicit campaign_id
            pass

    # reported_actor
    actor_raw = obj.get("reported_actor") or obj.get("actor") or obj.get("threat_actor") or ""
    reported_actor = str(actor_raw).strip() or None

    # If no campaign_id, fall back to actor as surrogate
    if not campaign_id and reported_actor:
        campaign_id = f"actor:{reported_actor.lower().replace(' ', '_')}"

    # TTPs
    ttps_raw = obj.get("ttps") or obj.get("techniques") or obj.get("disarm") or []
    if isinstance(ttps_raw, str):
        ttps_raw = [t.strip() for t in re.split(r"[,;]", ttps_raw) if t.strip()]
    ttps = [str(t) for t in ttps_raw if t]

    # target countries
    countries_raw = (
        obj.get("target_countries") or obj.get("countries") or obj.get("targets") or []
    )
    if isinstance(countries_raw, str):
        countries_raw = [c.strip() for c in re.split(r"[,;]", countries_raw) if c.strip()]
    target_countries = [str(c).upper()[:3] for c in countries_raw if c]

    # target sectors
    sectors_raw = obj.get("target_sectors") or obj.get("sectors") or []
    if isinstance(sectors_raw, str):
        sectors_raw = [s.strip() for s in re.split(r"[,;]", sectors_raw) if s.strip()]
    target_sectors = [str(s).lower() for s in sectors_raw if s]

    # channels
    channels_raw = obj.get("channels") or obj.get("platforms") or []
    if isinstance(channels_raw, str):
        channels_raw = [c.strip().lower() for c in re.split(r"[,;]", channels_raw) if c.strip()]
    channels = [str(c).lower() for c in channels_raw if c]

    # evidence_sources
    sources_raw = obj.get("evidence_sources") or obj.get("sources") or obj.get("urls") or []
    if isinstance(sources_raw, str):
        sources_raw = [sources_raw]
    evidence_sources = [str(s) for s in sources_raw if s]

    # dates
    first_seen = _parse_date(obj.get("first_seen") or obj.get("start_date") or obj.get("date"))
    last_seen = _parse_date(obj.get("last_seen") or obj.get("end_date") or first_seen)

    try:
        return Event(
            event_id=event_id,
            title=title,
            description=description,
            campaign_id=campaign_id,
            reported_actor=reported_actor,
            target_countries=target_countries,
            target_sectors=target_sectors,
            first_seen=first_seen,
            last_seen=last_seen,
            ttps=ttps,
            channels=channels,
            evidence_sources=evidence_sources,
            source_dataset="disinfox",
        )
    except Exception:
        return None


def _parse_date(value: Any) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m"):
        try:
            from datetime import datetime
            return datetime.strptime(s[:10], fmt).date()
        except ValueError:
            continue
    return None


def _fixture_fallback(interim_dir: Path, fallbacks_used: list[str]) -> list[Event]:
    """Generate fixture events and record fallback usage."""
    import sys
    from pathlib import Path as _Path

    # Ensure scripts/ is on path
    scripts_dir = _Path(__file__).parent.parent.parent.parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    from make_fixtures import make_fixtures
    events = make_fixtures(interim_dir)
    fallbacks_used.append("fixture mode: 20 synthetic scenario events used")
    _write_mapping_report(interim_dir, events, source="fixtures (fallback)")
    return events


def _write_mapping_report(out_dir: Path, events: list[Event], source: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    n_campaign = sum(1 for e in events if e.campaign_id)
    campaigns = sorted({e.campaign_id for e in events if e.campaign_id})

    lines = [
        "# DISINFOX → Event Schema Mapping Report",
        "",
        f"**Source**: `{source}`",
        f"**Total events**: {len(events)}",
        f"**Events with campaign_id**: {n_campaign}",
        f"**Unique campaigns**: {len(campaigns)}",
        "",
        "## Campaign list",
        "",
    ]
    for c in campaigns:
        count = sum(1 for e in events if e.campaign_id == c)
        lines.append(f"- `{c}`: {count} events")

    lines += [
        "",
        "## Mapping decisions",
        "",
        "| DISINFOX concept | Internal field | Note |",
        "|---|---|---|",
        "| incident id | `event_id` | Direct map |",
        "| title/name | `title` | Direct map |",
        "| description/content/summary | `description` | Embedding target |",
        "| campaign/campaign_id | `campaign_id` | **Ground truth source** |",
        "| reported_actor/actor | `reported_actor` | Context only, not used in eval |",
        "| ttps/techniques/disarm | `ttps` | TTP ID strings |",
        "| target_countries/countries | `target_countries` | Uppercased |",
        "| channels/platforms | `channels` | Lowercased |",
        "| evidence_sources/sources/urls | `evidence_sources` | URL list |",
        "| first_seen/start_date/date | `first_seen` | ISO date |",
        "| last_seen/end_date | `last_seen` | ISO date |",
        "",
    ]

    if "fixture" in source.lower():
        lines += [
            "## Fallback note",
            "",
            "DISINFOX data was unavailable. Fixture mode was activated.",
            "20 curated sample events are used for pipeline validation.",
            "These events cover Doppelganger (6), Ghostwriter (5), ",
            "Secondary Infektion (4), and unrelated (5) scenarios.",
            "Descriptions are original summaries based on public reports; no original text copied.",
            "",
            "**ζ=0 guard is especially important in this mode** since campaign_id is "
            "assigned from fixture data and not from live DISINFOX, making any actor-based "
            "heuristic a potential source of circular evaluation.",
        ]

    (out_dir / "mapping_report.md").write_text("\n".join(lines), encoding="utf-8")
