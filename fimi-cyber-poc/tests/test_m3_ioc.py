"""M3 tests: T1-T4, T11, T12."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fimicyber.ioc.extract import refang, defang, extract_iocs_from_event
from fimicyber.ioc.classify import classify_ioc
from fimicyber.ioc.confidence import compute_confidence
from fimicyber.ioc.synthetic import validate_all_reserved, generate_synthetic_iocs
from fimicyber.schema import Event, IOC


# ── T1: refang ───────────────────────────────────────────────────────────────

def test_T1_refang():
    result = refang("hxxps://bild[.]eu[.]com/x")
    assert result == "https://bild.eu.com/x"


def test_T1_refang_defang_roundtrip():
    original = "https://bild.eu.com/x"
    defanged = defang(original)
    refanged = refang(defanged)
    assert refanged == original


# ── T2: IPv4 validation ───────────────────────────────────────────────────────

def test_T2_invalid_ip_not_extracted():
    ev = Event(
        event_id="t2a", title="t", description="The IP 999.1.1.1 is fake."
    )
    iocs = extract_iocs_from_event(ev)
    ipv4_vals = [i.value for i in iocs if i.ioc_type == "ipv4"]
    assert "999.1.1.1" not in ipv4_vals


def test_T2_valid_rfc5737_ip_extracted():
    ev = Event(
        event_id="t2b",
        title="t",
        description="The C2 server at 203.0.113.7 was registered by the infrastructure team.",
    )
    iocs = extract_iocs_from_event(ev)
    ipv4_vals = [i.value for i in iocs if i.ioc_type == "ipv4"]
    assert "203.0.113.7" in ipv4_vals


# ── T3: classification ────────────────────────────────────────────────────────

def test_T3_phishing_domain_operational():
    cat, status = classify_ioc(
        value="evil-news.test",
        ioc_type="domain",
        context="phishing domain evil-news.test registered by the threat actor",
        evidence_sources=[],
    )
    assert cat == "OperationalIOC"


def test_T3_qurium_url_evidence_source():
    url = "https://qurium.org/alerts/doppelganger/"
    cat, status = classify_ioc(
        value=url,
        ioc_type="url",
        context="according to the report at https://qurium.org/alerts/doppelganger/",
        evidence_sources=[url],
    )
    assert cat == "EvidenceSourceURL"


# ── T4: IOCConfidence ─────────────────────────────────────────────────────────

def test_T4_confidence_golden():
    # C=(1.0, 1.0, 0.3, 0.8, 1.0)
    # Expected: 0.30*1.0 + 0.25*1.0 + 0.20*0.3 + 0.15*0.8 + 0.10*1.0 = 0.83
    from datetime import date

    components, confidence = compute_confidence(
        ioc_type="domain",           # C_type = 0.8
        category="OperationalIOC",
        context="phishing malware c2 domain registered infrastructure",  # ≥2 ops → C_context=1.0
        source_label="qurium.org",   # trusted → C_source=1.0
        n_sources=1,                 # C_corroboration=0.3
        event_first_seen=date(2022, 1, 1),
        event_last_seen=date(2022, 6, 1),
        ioc_first_seen=date(2022, 1, 1),  # overlap → C_freshness=1.0
        ioc_last_seen=date(2022, 6, 1),
    )
    assert abs(confidence - 0.83) < 1e-4, f"Expected ~0.83, got {confidence}"


# ── T11: Synthetic reserved ranges ───────────────────────────────────────────

def test_T11_synthetic_reserved_ranges(cfg):
    from fimicyber.loaders.disinfox import load_events

    fallbacks: list[str] = []
    events = load_events(cfg.data_dir / "raw", cfg, fallbacks)
    events = generate_synthetic_iocs(events, cfg)

    # validate_all_reserved raises if any violation
    validate_all_reserved(events)

    synth_iocs = [
        ioc for ev in events for ioc in ev.iocs if ioc.synthetic
    ]
    assert len(synth_iocs) > 0, "No synthetic IOCs generated"

    for ioc in synth_iocs:
        if ioc.ioc_type == "domain":
            assert ioc.value.endswith(".test"), ioc.value
        elif ioc.ioc_type == "ipv4":
            import ipaddress
            addr = ipaddress.IPv4Address(ioc.value)
            valid_nets = [
                ipaddress.IPv4Network("192.0.2.0/24"),
                ipaddress.IPv4Network("198.51.100.0/24"),
                ipaddress.IPv4Network("203.0.113.0/24"),
            ]
            assert any(addr in net for net in valid_nets), ioc.value
        elif ioc.ioc_type == "asn":
            asn_num = int(ioc.value.replace("AS", ""))
            assert 64496 <= asn_num <= 64511, ioc.value


# ── T12: Synthetic determinism ────────────────────────────────────────────────

def test_T12_synthetic_determinism(cfg):
    import hashlib, json

    from fimicyber.loaders.disinfox import load_events
    from fimicyber.ioc.synthetic import generate_synthetic_iocs

    fallbacks: list[str] = []

    # Run 1
    events1 = load_events(cfg.data_dir / "raw", cfg, fallbacks)
    events1 = generate_synthetic_iocs(events1, cfg)
    manifest1 = json.loads(
        (cfg.results_dir / "synthetic_manifest.json").read_text()
    )

    # Run 2 (fresh load)
    events2 = load_events(cfg.data_dir / "raw", cfg, fallbacks)
    events2 = generate_synthetic_iocs(events2, cfg)
    manifest2 = json.loads(
        (cfg.results_dir / "synthetic_manifest.json").read_text()
    )

    assert manifest1["sha256"] == manifest2["sha256"], (
        f"Manifest sha256 mismatch: {manifest1['sha256']} != {manifest2['sha256']}"
    )
