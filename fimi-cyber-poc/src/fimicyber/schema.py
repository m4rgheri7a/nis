"""Pydantic schemas — IOC and Event (spec 4.2)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class IOC(BaseModel):
    value: str
    ioc_type: Literal[
        "domain", "url", "ipv4", "email",
        "hash_md5", "hash_sha1", "hash_sha256",
        "ns", "asn", "account", "tg_channel",
    ]
    category: Literal[
        "EvidenceSourceURL", "PlatformContentURL",
        "OperationalIOC", "BenignReference",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    conf_components: dict[str, float] = Field(default_factory=dict)
    first_seen: date | None = None
    last_seen: date | None = None
    sources: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["candidate", "validated", "rejected", "needs_review"] = "candidate"
    synthetic: bool = False


class Event(BaseModel):
    event_id: str
    title: str
    description: str
    campaign_id: str | None = None
    campaign_id_source: Literal[
        "explicit", "actor_surrogate", "debunk_group", "fixture", "curated", "none"
    ] = "explicit"
    reported_actor: str | None = None
    target_countries: list[str] = Field(default_factory=list)
    target_sectors: list[str] = Field(default_factory=list)
    first_seen: date | None = None
    last_seen: date | None = None
    ttps: list[str] = Field(default_factory=list)
    channels: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    iocs: list[IOC] = Field(default_factory=list)
    source_dataset: str = "disinfox"
    evaluation_role: Literal["development", "reference", "holdout"] = "development"
    date_basis: Literal["observed", "published", "unknown"] = "observed"
    ai_artifact_signal: Literal["none", "suspected", "confirmed", "unknown"] = "unknown"
    provenance_signal: Literal[
        "present", "missing", "stripped", "inconsistent", "unknown"
    ] = "unknown"
    kill_chain_stage: Literal[
        "preparation",
        "content_creation",
        "seeding",
        "amplification",
        "exploitation",
        "unknown",
    ] = "unknown"
    llm_extracted: bool = False
    evidence_sentences: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)

