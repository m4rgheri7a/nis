"""Reconstructed case dossiers for leakage-controlled evidence structuring.

The generalisation benchmark keeps an analyst summary in ``Event.description``
and the published technical annex in a separate IOC CSV. An extraction
condition must read one document instead of the curated fields, so this module
rebuilds the text an analyst actually receives: the report summary plus a
defanged technical annex rendered from the same public IOC table.

Two invariants make the extraction conditions comparable to the curated oracle:

1. Curated analytic labels (``ttps``, ``channels``, ``target_sectors``,
   ``target_countries``, ``campaign_id``, ``reported_actor``) are never
   rendered into the dossier.
2. Actor and campaign names are removed from the rendered text, including from
   annex context sentences and report URLs, so no condition can read the answer
   out of its own input.

The annex is a reconstruction, not a verbatim copy of the source PDF: indicator
values, types and context sentences come from the public IOC tables already in
``data/external``. ``include_annex=False`` reproduces the summary-only input and
exists to quantify how much of the infrastructure signal depends on the annex.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fimicyber.attribution.taxonomy import load_actor_taxonomy
from fimicyber.ioc.extract import defang
from fimicyber.schema import Event

REDACTION = "[REDACTED-LABEL]"

_MIN_ALIAS_LEN = 3

_TYPE_LABELS: dict[str, str] = {
    "ipv4": "IPv4 address",
    "domain": "domain",
    "url": "URL",
    "email": "email address",
    "hash_md5": "MD5 hash",
    "hash_sha1": "SHA-1 hash",
    "hash_sha256": "SHA-256 hash",
    "account": "account handle",
    "tg_channel": "Telegram channel",
    "ns": "name server",
    "asn": "ASN",
}


@dataclass(frozen=True)
class AnnexRow:
    value: str
    ioc_type: str
    context: str


class LabelScrubber:
    """Redact actor, campaign, and report-slug labels from dossier text.

    ``hypotheses.py`` already nulls the narrative component when an actor name
    appears in an event description. That guard reacts to leakage; this one
    prevents it, and :func:`assert_no_labels` turns the guarantee into a test.
    """

    def __init__(self, labels: set[str]) -> None:
        self._labels = {label for label in labels if len(label) >= _MIN_ALIAS_LEN}
        # Longest first so "Storm-1516/Neva Flood" is consumed before "Storm-1516".
        ordered = sorted(self._labels, key=len, reverse=True)
        # Underscores and hyphens inside ids ("storm_1516") are not word
        # characters at the boundary, so match on non-alphanumeric edges.
        self._pattern = (
            re.compile(
                r"(?<![0-9A-Za-z])(?:" + "|".join(re.escape(l) for l in ordered) + r")(?![0-9A-Za-z])",
                re.IGNORECASE,
            )
            if ordered
            else None
        )

    @property
    def labels(self) -> set[str]:
        return set(self._labels)

    def scrub(self, text: str) -> str:
        if self._pattern is None or not text:
            return text
        return self._pattern.sub(REDACTION, text)

    def find(self, text: str) -> list[str]:
        """Return every label still present in ``text`` (empty means clean)."""
        if self._pattern is None or not text:
            return []
        return sorted({match.group(0) for match in self._pattern.finditer(text)})


# Fragments of composite labels that are ordinary vocabulary. Redacting these
# would damage the narrative text the extractor is supposed to read.
_GENERIC_FRAGMENTS = frozenset(
    {"internet", "research", "agency", "storm", "network", "flood", "operation"}
)


def build_label_scrubber(events: list[Event], cfg: Any) -> LabelScrubber:
    """Collect every alias, canonical name, campaign id, and id fragment to redact."""
    taxonomy = load_actor_taxonomy(cfg)
    labels: set[str] = set()

    taxonomy_path = cfg.data_dir / "curated" / cfg.attribution.get(
        "taxonomy_file", "actor_taxonomy.csv"
    )
    if taxonomy_path.exists():
        with taxonomy_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                # parent_actor_id is deliberately skipped: it encodes the
                # sponsoring state ("russia_state_linked"), and country names are
                # legitimate narrative content the extractor must still see.
                for column in ("alias", "canonical_name", "canonical_actor_id"):
                    value = (row.get(column) or "").strip().rstrip("*")
                    if value:
                        labels.add(value)

    for event in events:
        if event.campaign_id:
            labels.add(event.campaign_id)
        if event.reported_actor:
            labels.add(event.reported_actor)
            identity = taxonomy.resolve(event.reported_actor)
            if identity is not None:
                labels.add(identity.actor_id)
                labels.add(identity.display_name)
                labels.update(taxonomy.aliases_for(identity.actor_id))

    # Composite labels also leak through their parts: "Ghostwriter/UNC1151"
    # contains "Ghostwriter" and "spamouflage_dragonbridge" contains
    # "dragonbridge". Split only on the separators that join distinct aliases;
    # splitting on spaces would shred "Internet Research Agency" into words that
    # appear in ordinary prose. Hyphens need no split because the match boundary
    # is non-alphanumeric, so "prc-dragonbridge-elections" already hits
    # "dragonbridge".
    for label in list(labels):
        for fragment in re.split(r"[/_]+", label):
            fragment = fragment.strip().rstrip("*")
            if (
                len(fragment) >= _MIN_ALIAS_LEN
                and not fragment.isdigit()
                and fragment.casefold() not in _GENERIC_FRAGMENTS
            ):
                labels.add(fragment)

    return LabelScrubber(labels)


def load_annex_rows(cfg: Any) -> dict[str, list[AnnexRow]]:
    """Read the published IOC tables that stand in for the reports' technical annexes."""
    benchmark_cfg = cfg.attribution.get("generalization_benchmark", {})
    filenames = benchmark_cfg.get(
        "iocs_files", ["ghostwriter_iocs.csv", "multiactor_iocs.csv"]
    )
    rows: dict[str, list[AnnexRow]] = {}
    for filename in filenames:
        path: Path = cfg.data_dir / "external" / filename
        if not path.exists():
            continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                event_id = (row.get("event_id") or "").strip()
                value = (row.get("ioc_value") or "").strip()
                if not event_id or not value:
                    continue
                rows.setdefault(event_id, []).append(
                    AnnexRow(
                        value=value,
                        ioc_type=(row.get("ioc_type") or "domain").strip(),
                        context=(row.get("context") or "").strip(),
                    )
                )
    return rows


def build_case_dossier(
    event: Event,
    annex: list[AnnexRow],
    scrubber: LabelScrubber,
    include_annex: bool = True,
) -> str:
    """Render one leakage-controlled case document."""
    # Wording matters here. ``ioc.classify`` decides whether an observable is
    # adversary infrastructure or a citation by weighing operational keywords
    # ("indicator", "infrastructure", "c2") against source keywords ("report",
    # "source", "reference") in a +/-120 character window. An earlier annex
    # header read "indicators listed in the published report", whose bare
    # "report" outweighed the operational side and had real C2 addresses
    # classified as EvidenceSourceURL and dropped from the graph. The headings
    # below stay clear of source keywords.
    lines = [
        "CASE FILE (reconstructed from public reporting)",
        f"Title: {event.title}",
    ]
    if event.first_seen:
        window = str(event.first_seen)
        if event.last_seen and event.last_seen != event.first_seen:
            window = f"{event.first_seen} to {event.last_seen}"
        lines.append(f"Observation window: {window}")
    lines += ["", "Summary:", event.description]

    if include_annex and annex:
        lines += [
            "",
            "Technical annex - indicator and infrastructure list (defanged):",
        ]
        for row in annex:
            label = _TYPE_LABELS.get(row.ioc_type, row.ioc_type)
            entry = f"- {defang(row.value)} ({label})"
            if row.context:
                entry += f" - {row.context}"
            lines.append(entry)

    return scrubber.scrub("\n".join(lines))


def build_dossiers(
    events: list[Event],
    cfg: Any,
    include_annex: bool = True,
) -> dict[str, str]:
    """Build one dossier per event, keyed by ``event_id``."""
    scrubber = build_label_scrubber(events, cfg)
    annex_rows = load_annex_rows(cfg)
    return {
        event.event_id: build_case_dossier(
            event, annex_rows.get(event.event_id, []), scrubber, include_annex
        )
        for event in events
    }


def assert_no_labels(dossiers: dict[str, str], events: list[Event], cfg: Any) -> None:
    """Raise if any dossier still names an actor or campaign."""
    scrubber = build_label_scrubber(events, cfg)
    offenders = {
        event_id: found
        for event_id, text in dossiers.items()
        if (found := scrubber.find(text))
    }
    if offenders:
        raise ValueError(f"Actor/campaign labels leaked into dossiers: {offenders}")
