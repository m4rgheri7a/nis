"""Canonical actor identities and aliases for evaluation.

States, ecosystems, organisations, and campaigns remain separate levels. Only
aliases explicitly listed in the taxonomy are merged.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path


_NON_ID = re.compile(r"[^a-z0-9]+")


def normalise_alias(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return " ".join(value.strip().split()).casefold()


def _fallback_id(value: str) -> str:
    key = _NON_ID.sub("_", value.casefold()).strip("_")
    return key or "unmapped_actor"


@dataclass(frozen=True)
class ActorIdentity:
    actor_id: str
    display_name: str
    actor_level: str = "unknown"
    parent_actor_id: str = ""
    source_url: str = ""


class ActorTaxonomy:
    def __init__(self, aliases: dict[str, ActorIdentity] | None = None) -> None:
        self._aliases = aliases or {}
        self._identity_aliases: dict[str, set[str]] = {}
        for alias, identity in self._aliases.items():
            self._identity_aliases.setdefault(identity.actor_id, set()).add(alias)

    @classmethod
    def from_csv(cls, path: Path | None) -> "ActorTaxonomy":
        if path is None or not path.exists():
            return cls()
        aliases: dict[str, ActorIdentity] = {}
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                alias = normalise_alias(row.get("alias"))
                actor_id = (row.get("canonical_actor_id") or "").strip()
                display_name = (row.get("canonical_name") or "").strip()
                if not alias or not actor_id or not display_name:
                    continue
                aliases[alias] = ActorIdentity(
                    actor_id=actor_id,
                    display_name=display_name,
                    actor_level=(row.get("actor_level") or "unknown").strip(),
                    parent_actor_id=(row.get("parent_actor_id") or "").strip(),
                    source_url=(row.get("source_url") or "").strip(),
                )
        return cls(aliases)

    def resolve(self, label: str | None) -> ActorIdentity | None:
        alias = normalise_alias(label)
        if alias is None:
            return None
        if alias in self._aliases:
            return self._aliases[alias]
        display = " ".join(str(label).strip().split())
        return ActorIdentity(_fallback_id(alias), display)

    def aliases_for(self, actor_id: str) -> set[str]:
        return set(self._identity_aliases.get(actor_id, set()))


def load_actor_taxonomy(cfg: object) -> ActorTaxonomy:
    attribution = getattr(cfg, "attribution", {})
    configured = attribution.get("taxonomy_file", "actor_taxonomy.csv")
    path = Path(configured)
    if not path.is_absolute():
        path = getattr(cfg, "data_dir") / "curated" / path
    return ActorTaxonomy.from_csv(path)
