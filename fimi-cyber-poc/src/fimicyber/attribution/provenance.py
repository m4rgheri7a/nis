"""Evidence provenance export for public-source case records."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from fimicyber.schema import Event


def _record_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_manifest(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _manifest_match(evidence_id: str, rows: list[dict[str, str]]) -> dict[str, str]:
    matches = [row for row in rows if evidence_id.startswith(row.get("source_id", ""))]
    return max(matches, key=lambda row: len(row.get("source_id", ""))) if matches else {}


def build_evidence_provenance(events: list[Event], manifest_path: Path) -> pd.DataFrame:
    manifest = _load_manifest(manifest_path)
    output: list[dict[str, Any]] = []
    for event in events:
        event_dump = event.model_dump(mode="json", exclude={"iocs"})
        for evidence_id in event.evidence_ids:
            source = _manifest_match(evidence_id, manifest)
            output.append({
                "event_id": event.event_id,
                "evidence_kind": "event",
                "evidence_value": event.title,
                "evidence_id": evidence_id,
                "source_id": source.get("source_id", ""),
                "publisher": source.get("publisher", ""),
                "source_url": source.get("url", ""),
                "source_content_sha256": source.get("content_sha256", ""),
                "record_sha256": _record_hash(event_dump),
                "date_basis": event.date_basis,
            })
        for ioc in event.iocs:
            for evidence_id in ioc.evidence_ids:
                source = _manifest_match(evidence_id, manifest)
                output.append({
                    "event_id": event.event_id,
                    "evidence_kind": f"ioc:{ioc.ioc_type}",
                    "evidence_value": ioc.value,
                    "evidence_id": evidence_id,
                    "source_id": source.get("source_id", ""),
                    "publisher": source.get("publisher", ""),
                    "source_url": source.get("url", ""),
                    "source_content_sha256": source.get("content_sha256", ""),
                    "record_sha256": _record_hash(ioc.model_dump(mode="json")),
                    "date_basis": event.date_basis,
                })
    return pd.DataFrame(output)
