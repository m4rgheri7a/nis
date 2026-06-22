"""Synthetic IOC generator (spec 4.6).

ABSOLUTE RULE: Only RFC-reserved address spaces.
- Domains : .test TLD (RFC 2606)
- IPv4    : 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24 (RFC 5737)
- ASN     : 64496–64511 (RFC 5398)
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import random
import string
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from fimicyber.schema import Event, IOC

# ── Reserved ranges ────────────────────────────────────────────────────────

_RESERVED_NETS = [
    ipaddress.IPv4Network("192.0.2.0/24"),
    ipaddress.IPv4Network("198.51.100.0/24"),
    ipaddress.IPv4Network("203.0.113.0/24"),
]
_ASN_MIN = 64496
_ASN_MAX = 64511
_DOMAIN_TLD = ".test"


def _validate_reserved(ioc: IOC) -> None:
    """Raise ValueError if synthetic IOC violates reserved-range rule."""
    if ioc.ioc_type == "domain":
        if not ioc.value.endswith(_DOMAIN_TLD):
            raise ValueError(f"Synthetic domain not .test: {ioc.value!r}")
    elif ioc.ioc_type == "ipv4":
        addr = ipaddress.IPv4Address(ioc.value)
        if not any(addr in net for net in _RESERVED_NETS):
            raise ValueError(f"Synthetic IP not in RFC5737 block: {ioc.value!r}")
    elif ioc.ioc_type == "asn":
        asn_num = int(ioc.value.replace("AS", "").replace("as", ""))
        if not (_ASN_MIN <= asn_num <= _ASN_MAX):
            raise ValueError(f"Synthetic ASN out of range: {ioc.value!r}")
    elif ioc.ioc_type == "ns":
        if not ioc.value.endswith(_DOMAIN_TLD):
            raise ValueError(f"Synthetic NS not .test: {ioc.value!r}")
    elif ioc.ioc_type == "email":
        if not ioc.value.endswith(_DOMAIN_TLD):
            raise ValueError(f"Synthetic email not @*.test: {ioc.value!r}")


# ── Generators ─────────────────────────────────────────────────────────────

def _rand_str(rng: random.Random, length: int = 8) -> str:
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def _gen_domain(rng: random.Random) -> str:
    return f"{_rand_str(rng, 6)}-{_rand_str(rng, 4)}{_DOMAIN_TLD}"


def _gen_url(rng: random.Random, domain: str) -> str:
    path = _rand_str(rng, 5)
    return f"http://{domain}/{path}"


def _gen_ipv4(rng: random.Random) -> str:
    net = rng.choice(_RESERVED_NETS)
    # random host within /24 (avoid .0 and .255)
    host = rng.randint(1, 254)
    base = str(net.network_address).rsplit(".", 1)[0]
    return f"{base}.{host}"


def _gen_hash_sha256(rng: random.Random) -> str:
    return "".join(rng.choices("0123456789abcdef", k=64))


def _gen_email(rng: random.Random) -> str:
    return f"{_rand_str(rng, 5)}@{_rand_str(rng, 6)}{_DOMAIN_TLD}"


def _gen_ns(rng: random.Random) -> str:
    return f"ns1.{_rand_str(rng, 6)}{_DOMAIN_TLD}"


def _gen_asn(rng: random.Random) -> str:
    return f"AS{rng.randint(_ASN_MIN, _ASN_MAX)}"


_TYPE_GENERATORS = {
    "domain": _gen_domain,
    "url": _gen_url,  # special: takes domain too
    "ipv4": _gen_ipv4,
    "hash_sha256": _gen_hash_sha256,
    "email": _gen_email,
}

_IOC_FROM_CONF = {
    "C_context": 0.7,
    "C_source": 0.8,
    "C_corroboration": 0.3,
    "C_type": 0.7,  # will be overridden by type
    "C_freshness": 1.0,
}

_TYPE_C_TYPE = {
    "domain": 0.8, "url": 1.0, "ipv4": 0.4,
    "hash_sha256": 1.0, "email": 0.9,
}


def _make_ioc(
    value: str,
    ioc_type: str,
    first_seen: date | None = None,
    last_seen: date | None = None,
) -> IOC:
    from fimicyber.ioc.confidence import compute_confidence
    components, confidence = compute_confidence(
        ioc_type=ioc_type,
        category="OperationalIOC",
        context="phishing domain registered infrastructure",  # ≥2 op keywords
        source_label="synthetic",
        n_sources=1,
        event_first_seen=first_seen,
        event_last_seen=last_seen,
        ioc_first_seen=first_seen,
        ioc_last_seen=last_seen,
    )
    # Override C_source per spec 4.5
    components["C_source"] = 0.8
    components["C_corroboration"] = 0.3
    confidence = round(
        0.30 * components["C_context"]
        + 0.25 * 0.8
        + 0.20 * 0.3
        + 0.15 * components["C_type"]
        + 0.10 * components["C_freshness"],
        6,
    )
    ioc = IOC(
        value=value,
        ioc_type=ioc_type,
        category="OperationalIOC",
        confidence=confidence,
        conf_components=components,
        sources=["synthetic"],
        status="candidate",
        synthetic=True,
    )
    _validate_reserved(ioc)
    return ioc


# ── Main entry point ────────────────────────────────────────────────────────

def generate_synthetic_iocs(events: list[Event], cfg: Any) -> list[Event]:
    """Attach synthetic IOCs to events per spec 4.6. Returns modified events."""
    sc = cfg.synthetic
    default = sc["default"]
    coverage: float = default["coverage"]
    noise_ratio: float = default["noise_ratio"]
    jitter_days: int = default["temporal_jitter_days"]
    ns_link_prob: float = default["ns_link_prob"]
    type_mix: dict[str, float] = default["type_mix"]
    base_seed: int = int(sc.get("base_seed", 42))

    # Validate config constraints (spec 4.6)
    if coverage >= 1.0:
        raise ValueError("synthetic coverage must be < 1.0")
    if noise_ratio <= 0.0:
        raise ValueError("synthetic noise_ratio must be > 0.0 in default config")

    rng = random.Random(base_seed)

    # Build campaign → event map (only campaign_id ≠ None, ≥2 events)
    from collections import defaultdict
    camp_map: dict[str, list[int]] = defaultdict(list)
    for idx, ev in enumerate(events):
        if ev.campaign_id:
            camp_map[ev.campaign_id].append(idx)

    manifest: dict[str, Any] = {
        "seed": base_seed,
        "coverage": coverage,
        "noise_ratio": noise_ratio,
        "jitter_days": jitter_days,
        "injections": [],
        "noise_injections": [],
    }

    # Shared IOC pools per campaign
    camp_shared_iocs: dict[str, list[IOC]] = {}
    camp_shared_ns: dict[str, str | None] = {}

    for camp_id, idxs in camp_map.items():
        if len(idxs) < 2:
            continue

        # Pick how many IOCs to share (weighted by type_mix, ~5 per campaign)
        n_shared = max(2, rng.randint(3, 7))
        pool: list[IOC] = _generate_pool(rng, n_shared, type_mix, ns_link_prob)
        camp_shared_iocs[camp_id] = pool

        # Optional shared NS
        if rng.random() < ns_link_prob:
            ns_val = _gen_ns(rng)
            camp_shared_ns[camp_id] = ns_val
        else:
            camp_shared_ns[camp_id] = None

    # Noise pool: shared low-quality IOCs (shared hosting IP + short-URL domain)
    noise_pool = _generate_noise_pool(rng)

    # Assign IOCs to events
    for camp_id, idxs in camp_map.items():
        if len(idxs) < 2 or camp_id not in camp_shared_iocs:
            continue

        pool = camp_shared_iocs[camp_id]
        ns_val = camp_shared_ns.get(camp_id)

        for idx in idxs:
            ev = events[idx]
            # Sample `coverage` fraction of pool
            n_assign = max(1, int(len(pool) * coverage))
            chosen = rng.sample(pool, min(n_assign, len(pool)))

            # Add temporal jitter
            dated_iocs = _apply_jitter(chosen, ev, jitter_days, rng)

            ev.iocs.extend(dated_iocs)

            # Add NS IOC if applicable
            if ns_val:
                ns_ioc = _make_ioc(ns_val, "ns", ev.first_seen, ev.last_seen)
                ev.iocs.append(ns_ioc)

            manifest["injections"].append(
                {
                    "campaign": camp_id,
                    "event_id": ev.event_id,
                    "ioc_values": [i.value for i in dated_iocs],
                    "ns": ns_val,
                }
            )

    # Noise injection
    all_idxs = list(range(len(events)))
    n_noise = max(1, int(len(events) * noise_ratio))
    noise_targets = rng.sample(all_idxs, min(n_noise, len(all_idxs)))

    for idx in noise_targets:
        ev = events[idx]
        noise_iocs = _apply_jitter(noise_pool, ev, jitter_days, rng)
        ev.iocs.extend(noise_iocs)
        manifest["noise_injections"].append(
            {
                "event_id": ev.event_id,
                "ioc_values": [i.value for i in noise_iocs],
            }
        )

    # Save manifest
    _save_manifest(manifest, cfg)

    return events


def _generate_pool(
    rng: random.Random,
    n: int,
    type_mix: dict[str, float],
    ns_prob: float,
) -> list[IOC]:
    """Generate a pool of n synthetic IOCs according to type_mix."""
    types = list(type_mix.keys())
    weights = [type_mix[t] for t in types]
    chosen_types = rng.choices(types, weights=weights, k=n)

    iocs: list[IOC] = []
    generated_domains: list[str] = []

    for itype in chosen_types:
        if itype == "domain":
            domain = _gen_domain(rng)
            generated_domains.append(domain)
            ioc = _make_ioc(domain, "domain")
        elif itype == "url":
            if generated_domains:
                dom = rng.choice(generated_domains)
            else:
                dom = _gen_domain(rng)
            ioc = _make_ioc(_gen_url(rng, dom), "url")
        elif itype == "ipv4":
            ioc = _make_ioc(_gen_ipv4(rng), "ipv4")
        elif itype == "hash_sha256":
            ioc = _make_ioc(_gen_hash_sha256(rng), "hash_sha256")
        elif itype == "email":
            ioc = _make_ioc(_gen_email(rng), "email")
        else:
            continue
        iocs.append(ioc)

    return iocs


def _generate_noise_pool(rng: random.Random) -> list[IOC]:
    """Low-quality shared infrastructure IOCs for noise injection."""
    shared_ip = _gen_ipv4(rng)
    shared_domain = _gen_domain(rng)
    return [
        _make_ioc(shared_ip, "ipv4"),
        _make_ioc(shared_domain, "domain"),
    ]


def _apply_jitter(
    iocs: list[IOC],
    ev: Event,
    jitter_days: int,
    rng: random.Random,
) -> list[IOC]:
    """Return new IOC instances with temporally jittered dates."""
    result: list[IOC] = []
    base = ev.first_seen

    for ioc in iocs:
        delta = rng.randint(-jitter_days, jitter_days)
        if base is not None:
            fs = base + timedelta(days=delta)
            ls = fs + timedelta(days=rng.randint(0, 30))
        else:
            fs = ls = None

        new_ioc = ioc.model_copy(update={"first_seen": fs, "last_seen": ls})
        result.append(new_ioc)

    return result


def _save_manifest(manifest: dict[str, Any], cfg: Any) -> None:
    out_path = cfg.results_dir / "synthetic_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute sha256 of deterministic content
    content = json.dumps(manifest, sort_keys=True, default=str)
    manifest["sha256"] = hashlib.sha256(content.encode()).hexdigest()

    out_path.write_text(
        json.dumps(manifest, indent=2, default=str, sort_keys=True),
        encoding="utf-8",
    )


def validate_all_reserved(events: list[Event]) -> None:
    """Raise if any synthetic IOC violates the reserved-range rule (T11)."""
    for ev in events:
        for ioc in ev.iocs:
            if ioc.synthetic:
                _validate_reserved(ioc)
