# Curated real IOC case: Doppelganger 2022

This folder adds one small public-IOC-backed case to the PoC. The initial case is based on Qurium and EU DisinfoLab's public Doppelganger investigation.

Files:
- `curated_events.jsonl`: six normalized Events for media-clone infrastructure clusters.
- `real_iocs.csv`: validated non-synthetic OperationalIOC values attached to those Events.
- `ioc_relations.csv`: public infrastructure relations such as domain-to-nameserver, domain-to-IP, and redirect paths.

Primary sources:
- Qurium, "Under the hood of a Doppelganger": https://www.qurium.org/alerts/russia/under-the-hood-of-a-doppelganger/
- EU DisinfoLab, "Doppelganger - Media clones serving Russian propaganda": https://www.disinfo.eu/doppelganger/

These IOCs are used as public research evidence for graph/path validation, not as a live blocklist.
