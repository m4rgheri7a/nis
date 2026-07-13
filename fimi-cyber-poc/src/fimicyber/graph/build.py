"""Evidence graph builder (spec 6.1).

Nodes: Event, Campaign, Actor, IOC, TTP, Channel, EvidenceSource, Domain, IP, NS, ASN
Edges: LINKED_TO, USES, DISTRIBUTED_ON, SUPPORTED_BY,
       PART_OF, REPORTED_ATTRIBUTION, RESOLVES_TO, USES_NS,
       BELONGS_TO_ASN, REDIRECTS_TO, REGISTERED_AT
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import networkx as nx

from fimicyber.schema import Event, IOC


def _node_id(ntype: str, value: str) -> str:
    return f"{ntype}:{value}"


def build_graph(events: list[Event], cfg: Any) -> nx.Graph:
    G = nx.Graph()

    # ── Add Event nodes ────────────────────────────────────────────────────
    for ev in events:
        G.add_node(
            _node_id("Event", ev.event_id),
            ntype="Event",
            event_id=ev.event_id,
            campaign_id=ev.campaign_id,
            first_seen=str(ev.first_seen or ""),
            last_seen=str(ev.last_seen or ""),
        )

    # ── Add per-event nodes & edges ────────────────────────────────────────
    for ev in events:
        ev_node = _node_id("Event", ev.event_id)

        # Campaign and actor labels are reference context only. Scoring never
        # traverses these nodes, preventing ground-truth leakage into FCLS.
        if ev.campaign_id:
            campaign_node = _node_id("Campaign", ev.campaign_id)
            G.add_node(campaign_node, ntype="Campaign", value=ev.campaign_id)
            G.add_edge(
                ev_node, campaign_node,
                etype="PART_OF",
                weight=1.0,
                confidence=1.0,
                first_seen=str(ev.first_seen or ""),
                last_seen=str(ev.last_seen or ""),
                explanation=f"Event {ev.event_id} is reported in campaign {ev.campaign_id}",
                reference_only=True,
            )

        if ev.reported_actor:
            actor_node = _node_id("Actor", ev.reported_actor)
            G.add_node(actor_node, ntype="Actor", value=ev.reported_actor)
            G.add_edge(
                ev_node, actor_node,
                etype="REPORTED_ATTRIBUTION",
                weight=1.0,
                confidence=1.0,
                first_seen=str(ev.first_seen or ""),
                last_seen=str(ev.last_seen or ""),
                explanation=f"Public source reports actor label {ev.reported_actor}",
                reference_only=True,
            )

        # TTPs
        for ttp in ev.ttps:
            ttp_node = _node_id("TTP", ttp)
            G.add_node(ttp_node, ntype="TTP", value=ttp)
            G.add_edge(
                ev_node, ttp_node,
                etype="USES",
                weight=1.0,
                confidence=1.0,
                first_seen=str(ev.first_seen or ""),
                last_seen=str(ev.last_seen or ""),
                explanation=f"Event {ev.event_id} uses TTP {ttp}",
            )

        # Channels
        for ch in ev.channels:
            ch_node = _node_id("Channel", ch)
            G.add_node(ch_node, ntype="Channel", value=ch)
            G.add_edge(
                ev_node, ch_node,
                etype="DISTRIBUTED_ON",
                weight=1.0,
                confidence=1.0,
                first_seen=str(ev.first_seen or ""),
                last_seen=str(ev.last_seen or ""),
                explanation=f"Event {ev.event_id} distributed on {ch}",
            )

        # Evidence sources
        for src in ev.evidence_sources:
            src_node = _node_id("EvidenceSource", src)
            G.add_node(src_node, ntype="EvidenceSource", value=src)
            G.add_edge(
                ev_node, src_node,
                etype="SUPPORTED_BY",
                weight=1.0,
                confidence=1.0,
                first_seen="",
                last_seen="",
                explanation=f"Event {ev.event_id} supported by {src}",
            )

        # OperationalIOCs only
        for ioc in ev.iocs:
            if ioc.category != "OperationalIOC":
                continue

            ioc_node = _node_id("IOC", ioc.value)
            G.add_node(
                ioc_node,
                ntype="IOC",
                value=ioc.value,
                ioc_type=ioc.ioc_type,
                confidence=ioc.confidence,
                synthetic=ioc.synthetic,
                first_seen=str(ioc.first_seen or ""),
                last_seen=str(ioc.last_seen or ""),
            )
            G.add_edge(
                ev_node, ioc_node,
                etype="LINKED_TO",
                weight=ioc.confidence,
                confidence=ioc.confidence,
                first_seen=str(ioc.first_seen or ""),
                last_seen=str(ioc.last_seen or ""),
                explanation=f"Event {ev.event_id} linked to IOC {ioc.value}",
            )

            # Add infra sub-nodes for domain and IP
            _add_infra_nodes(G, ioc, synthetic=ioc.synthetic)

    # ── Load curated IOC relations ─────────────────────────────────────────
    _load_curated_relations(G, cfg.data_dir / "curated" / "ioc_relations.csv")

    return G


def _add_infra_nodes(G: nx.Graph, ioc: IOC, synthetic: bool) -> None:
    """Add domain/IP/NS/ASN infra nodes derived from an IOC."""
    ioc_node = _node_id("IOC", ioc.value)
    conf = 0.85 if synthetic else 0.9

    if ioc.ioc_type in ("domain", "url"):
        domain = _extract_domain(ioc.value)
        if domain:
            dom_node = _node_id("Domain", domain)
            G.add_node(dom_node, ntype="Domain", value=domain)
            if dom_node != ioc_node:
                G.add_edge(
                    ioc_node, dom_node,
                    etype="RESOLVES_TO",
                    weight=conf,
                    confidence=conf,
                    first_seen=str(ioc.first_seen or ""),
                    last_seen=str(ioc.last_seen or ""),
                    explanation=f"IOC {ioc.value} resolves to domain {domain}",
                )

    if ioc.ioc_type == "ipv4":
        ip_node = _node_id("IP", ioc.value)
        G.add_node(ip_node, ntype="IP", value=ioc.value)
        if ip_node != ioc_node:
            G.add_edge(
                ioc_node, ip_node,
                etype="RESOLVES_TO",
                weight=conf,
                confidence=conf,
                first_seen="", last_seen="",
                explanation=f"IOC {ioc.value} is IP",
            )

    if ioc.ioc_type == "ns":
        ns_node = _node_id("NS", ioc.value)
        G.add_node(ns_node, ntype="NS", value=ioc.value)
        if ns_node != ioc_node:
            G.add_edge(
                ioc_node, ns_node,
                etype="USES_NS",
                weight=conf,
                confidence=conf,
                first_seen="", last_seen="",
                explanation=f"IOC {ioc.value} uses NS",
            )

    if ioc.ioc_type == "asn":
        asn_node = _node_id("ASN", ioc.value)
        G.add_node(asn_node, ntype="ASN", value=ioc.value)
        if asn_node != ioc_node:
            G.add_edge(
                ioc_node, asn_node,
                etype="BELONGS_TO_ASN",
                weight=conf,
                confidence=conf,
                first_seen="", last_seen="",
                explanation=f"IOC {ioc.value} belongs to ASN",
            )


def _extract_domain(value: str) -> str | None:
    value = re.sub(r"^https?://", "", value)
    domain = value.split("/")[0].split(":")[0].lower()
    return domain if "." in domain else None


def _load_curated_relations(G: nx.Graph, path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row.get("src_value", "").strip()
            dst = row.get("dst_value", "").strip()
            relation = row.get("relation", "").strip().upper()
            conf = float(row.get("confidence", "0.9") or "0.9")

            if not src or not dst:
                continue

            # Determine node types
            src_ntype = _infer_ntype(src)
            dst_ntype = _infer_ntype(dst)

            src_node = _node_id(src_ntype, src)
            dst_node = _node_id(dst_ntype, dst)

            if src_node not in G:
                G.add_node(src_node, ntype=src_ntype, value=src)
            if dst_node not in G:
                G.add_node(dst_node, ntype=dst_ntype, value=dst)

            G.add_edge(
                src_node, dst_node,
                etype=relation,
                weight=conf,
                confidence=conf,
                first_seen="",
                last_seen="",
                explanation=f"Curated: {src} {relation} {dst}",
            )


def _infer_ntype(value: str) -> str:
    import re, ipaddress
    try:
        ipaddress.IPv4Address(value)
        return "IP"
    except ValueError:
        pass
    if value.startswith("AS") and value[2:].isdigit():
        return "ASN"
    lower = value.lower()
    if lower.startswith("ns") or lower.endswith(".ns.cloudflare.com"):
        return "NS"
    return "Domain"


# ── Infra subgraph helper (used by ioc_score) ──────────────────────────────

_INFRA_NTYPES: frozenset[str] = frozenset({"IOC", "Domain", "IP", "NS", "ASN"})


def infra_subgraph(G: nx.Graph) -> nx.Graph:
    """Return subgraph containing only infrastructure nodes (no Event/TTP/Channel/etc.)."""
    infra_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") in _INFRA_NTYPES]
    return G.subgraph(infra_nodes).copy()

