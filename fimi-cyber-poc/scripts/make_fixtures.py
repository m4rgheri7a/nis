#!/usr/bin/env python3
"""Generate sample fixture events (fallback when DISINFOX data unavailable).

Produces 20 events across 4 campaigns:
  - doppelganger (6 events)
  - ghostwriter   (5 events)
  - secondary_infektion (4 events)
  - unrelated      (5 events — no campaign_id)

Descriptions are original summaries based on public threat-intelligence reports;
no original text is copied.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fimicyber.schema import Event

_EVENTS: list[dict] = [
    # ── Doppelganger campaign (6) ────────────────────────────────────────
    {
        "event_id": "DG-001",
        "title": "Doppelganger clones major European news portals",
        "description": (
            "A coordinated pro-Kremlin influence operation created spoofed versions of "
            "Der Spiegel, Le Monde, and Bild websites to disseminate fabricated articles "
            "supporting Russian narratives about Ukraine. The infrastructure used typosquatting "
            "domains registered via anonymous providers. Content was amplified by networks of "
            "inauthentic social media accounts."
        ),
        "campaign_id": "doppelganger",
        "reported_actor": "CopyCop / Social Design Agency",
        "target_countries": ["DE", "FR", "UA"],
        "target_sectors": ["media", "government"],
        "first_seen": "2022-07-01",
        "last_seen": "2022-09-30",
        "ttps": ["T0008", "T0022", "T0045"],
        "channels": ["twitter", "facebook", "telegram"],
        "evidence_sources": [
            "https://www.qurium.org/alerts/under-the-hood-of-a-doppelganger/",
            "https://disinfo.eu/2022/08/doppelganger-cloning-european-media/",
        ],
    },
    {
        "event_id": "DG-002",
        "title": "Doppelganger targets French election discourse",
        "description": (
            "Spoofed French news outlet pages published fabricated stories alleging "
            "corruption among pro-EU candidates. The operation used redirect chains through "
            "multiple intermediary domains to obscure the true destination. Attribution markers "
            "in the HTML metadata matched previously identified Doppelganger infrastructure."
        ),
        "campaign_id": "doppelganger",
        "reported_actor": "Social Design Agency",
        "target_countries": ["FR"],
        "target_sectors": ["elections", "politics"],
        "first_seen": "2022-09-01",
        "last_seen": "2022-12-15",
        "ttps": ["T0008", "T0046", "T0049"],
        "channels": ["facebook", "instagram"],
        "evidence_sources": [
            "https://disinfo.eu/2022/10/doppelganger-france-elections/",
        ],
    },
    {
        "event_id": "DG-003",
        "title": "Doppelganger German-language disinformation wave",
        "description": (
            "A wave of fake articles mimicking German-language media falsely claimed NATO "
            "troops had caused civilian casualties. Domains shared nameservers with previously "
            "flagged Doppelganger infrastructure. The content was pushed via German-language "
            "Telegram channels with large subscriber counts."
        ),
        "campaign_id": "doppelganger",
        "reported_actor": "Social Design Agency",
        "target_countries": ["DE", "AT", "CH"],
        "target_sectors": ["military", "government"],
        "first_seen": "2023-01-10",
        "last_seen": "2023-04-20",
        "ttps": ["T0008", "T0022"],
        "channels": ["telegram"],
        "evidence_sources": [
            "https://www.qurium.org/alerts/doppelganger-german-2023/",
        ],
    },
    {
        "event_id": "DG-004",
        "title": "Doppelganger repurposes Italian media clones",
        "description": (
            "Italian-language clones of La Repubblica and Corriere della Sera were used to "
            "spread false claims about Italian government support for Ukraine. The spoofed sites "
            "used the same bulletproof hosting ASN observed in earlier Doppelganger campaigns. "
            "Inauthentic amplifier accounts on Facebook shared links at coordinated intervals."
        ),
        "campaign_id": "doppelganger",
        "reported_actor": "CopyCop",
        "target_countries": ["IT"],
        "target_sectors": ["elections", "media"],
        "first_seen": "2023-02-01",
        "last_seen": "2023-05-30",
        "ttps": ["T0008", "T0045"],
        "channels": ["facebook", "twitter"],
        "evidence_sources": [
            "https://disinfo.eu/2023/03/doppelganger-italy/",
        ],
    },
    {
        "event_id": "DG-005",
        "title": "Doppelganger UK-targeting expansion",
        "description": (
            "Newly registered domains impersonating BBC and The Guardian were discovered "
            "hosting fabricated articles about UK foreign policy failures. DNS analysis revealed "
            "the domains resolved to the same IP block used in prior Doppelganger clusters. "
            "The campaign coincided with a NATO ministerial meeting."
        ),
        "campaign_id": "doppelganger",
        "reported_actor": "Social Design Agency",
        "target_countries": ["GB"],
        "target_sectors": ["government", "media"],
        "first_seen": "2023-07-01",
        "last_seen": "2023-09-01",
        "ttps": ["T0008", "T0022", "T0046"],
        "channels": ["twitter", "facebook"],
        "evidence_sources": [
            "https://www.qurium.org/alerts/doppelganger-uk-2023/",
        ],
    },
    {
        "event_id": "DG-006",
        "title": "Doppelganger infrastructure refresh detected",
        "description": (
            "Following public attribution, operators migrated to a new hosting provider "
            "while reusing the same domain-generation pattern and nameserver configuration. "
            "WHOIS records showed coordinated registration dates. The refreshed infrastructure "
            "immediately began serving cloned media pages targeting Poland and Baltic states."
        ),
        "campaign_id": "doppelganger",
        "reported_actor": "Social Design Agency",
        "target_countries": ["PL", "EE", "LV", "LT"],
        "target_sectors": ["government", "elections"],
        "first_seen": "2023-10-01",
        "last_seen": "2024-01-15",
        "ttps": ["T0008", "T0022", "T0049"],
        "channels": ["facebook", "telegram"],
        "evidence_sources": [
            "https://disinfo.eu/2023/11/doppelganger-new-infra/",
        ],
    },

    # ── Ghostwriter campaign (5) ─────────────────────────────────────────
    {
        "event_id": "GW-001",
        "title": "Ghostwriter hacks Baltic news portals to plant fake stories",
        "description": (
            "Attributed to UNC1151, Ghostwriter compromised the content management systems "
            "of Lithuanian and Latvian news websites to inject fabricated articles claiming "
            "NATO soldiers committed crimes against locals. Mandiant analysis found overlapping "
            "C2 infrastructure with Belarus state cyber activity."
        ),
        "campaign_id": "ghostwriter",
        "reported_actor": "UNC1151",
        "target_countries": ["LT", "LV", "EE"],
        "target_sectors": ["military", "government"],
        "first_seen": "2020-11-01",
        "last_seen": "2021-06-30",
        "ttps": ["T0046", "T0057", "T0084"],
        "channels": ["web", "facebook"],
        "evidence_sources": [
            "https://www.mandiant.com/resources/ghostwriter-influence-campaign",
        ],
    },
    {
        "event_id": "GW-002",
        "title": "Ghostwriter forges official documents to undermine NATO",
        "description": (
            "Forged letters purportedly from NATO officials were published on compromised "
            "government and news websites in Poland and Ukraine. The documents falsely described "
            "plans to withdraw alliance support. The same phishing infrastructure used to gain "
            "CMS access was traced to prior UNC1151 spear-phishing campaigns."
        ),
        "campaign_id": "ghostwriter",
        "reported_actor": "UNC1151",
        "target_countries": ["PL", "UA"],
        "target_sectors": ["government", "military"],
        "first_seen": "2021-03-01",
        "last_seen": "2021-09-01",
        "ttps": ["T0046", "T0057"],
        "channels": ["web"],
        "evidence_sources": [
            "https://www.mandiant.com/resources/ghostwriter-unc1151",
        ],
    },
    {
        "event_id": "GW-003",
        "title": "Ghostwriter targets German election with fake Bundeswehr content",
        "description": (
            "Fake social media posts attributed to Bundeswehr accounts alleged soldiers "
            "were experiencing low morale and refusing deployments. The accounts used "
            "profile pictures harvested from real German military personnel. Server logs "
            "linked the campaign to Ghostwriter-associated IPs."
        ),
        "campaign_id": "ghostwriter",
        "reported_actor": "UNC1151",
        "target_countries": ["DE"],
        "target_sectors": ["military", "elections"],
        "first_seen": "2021-07-01",
        "last_seen": "2021-09-26",
        "ttps": ["T0022", "T0057"],
        "channels": ["facebook", "twitter"],
        "evidence_sources": [
            "https://www.mandiant.com/resources/ghostwriter-germany",
        ],
    },
    {
        "event_id": "GW-004",
        "title": "Ghostwriter escalates Ukrainian disinformation before invasion",
        "description": (
            "In the weeks before Russia's full-scale invasion, Ghostwriter-linked accounts "
            "published fabricated Ukrainian government statements announcing capitulation. "
            "Websites mimicking official Ukrainian government portals were registered using "
            "the same registrar pattern as prior GW infrastructure."
        ),
        "campaign_id": "ghostwriter",
        "reported_actor": "UNC1151",
        "target_countries": ["UA", "PL"],
        "target_sectors": ["government"],
        "first_seen": "2022-01-15",
        "last_seen": "2022-02-28",
        "ttps": ["T0046", "T0084", "T0008"],
        "channels": ["telegram", "web"],
        "evidence_sources": [
            "https://www.mandiant.com/resources/ghostwriter-ukraine-2022",
        ],
    },
    {
        "event_id": "GW-005",
        "title": "Ghostwriter resurfaces with Polish-language disinformation",
        "description": (
            "A renewed Ghostwriter cluster targeted Polish-language audiences with fabricated "
            "quotes from Polish officials allegedly calling for reduced support to Ukraine. "
            "Domain registration patterns and hosting providers matched previously attributed "
            "Ghostwriter infrastructure with high confidence."
        ),
        "campaign_id": "ghostwriter",
        "reported_actor": "UNC1151",
        "target_countries": ["PL"],
        "target_sectors": ["government", "elections"],
        "first_seen": "2022-10-01",
        "last_seen": "2023-01-31",
        "ttps": ["T0046", "T0008"],
        "channels": ["facebook", "web"],
        "evidence_sources": [
            "https://disinfo.eu/2022/11/ghostwriter-poland/",
        ],
    },

    # ── Secondary Infektion (4) ──────────────────────────────────────────
    {
        "event_id": "SI-001",
        "title": "Secondary Infektion fabricates leak documents on EU Forum",
        "description": (
            "EU DisinfoLab uncovered a network of fake personas on European online forums "
            "that posted fabricated intelligence leak documents. The accounts were created in "
            "coordinated batches with similar email patterns. Content targeted EU-Russia "
            "sanctions discussions."
        ),
        "campaign_id": "secondary_infektion",
        "reported_actor": "Secondary Infektion",
        "target_countries": ["EU", "DE", "FR"],
        "target_sectors": ["government", "politics"],
        "first_seen": "2019-01-01",
        "last_seen": "2020-06-01",
        "ttps": ["T0022", "T0023"],
        "channels": ["reddit", "web"],
        "evidence_sources": [
            "https://eu.disinfo.eu/secondary-infektion/",
        ],
    },
    {
        "event_id": "SI-002",
        "title": "Secondary Infektion pushes anti-Ukraine narratives on Avaaz petition sites",
        "description": (
            "Fake petition campaigns on third-party platforms collected signatures against "
            "EU arms supplies to Ukraine. The petition texts contained linguistic markers "
            "consistent with machine-translated Russian-origin content. Account registration "
            "IP addresses clustered in Russian hosting blocks."
        ),
        "campaign_id": "secondary_infektion",
        "reported_actor": "Secondary Infektion",
        "target_countries": ["UA", "EU"],
        "target_sectors": ["politics", "government"],
        "first_seen": "2019-06-01",
        "last_seen": "2020-12-01",
        "ttps": ["T0023", "T0049"],
        "channels": ["web"],
        "evidence_sources": [
            "https://graphika.com/reports/secondary-infektion/",
        ],
    },
    {
        "event_id": "SI-003",
        "title": "Secondary Infektion spoofs European think-tank publications",
        "description": (
            "Fake reports impersonating European think-tanks were distributed via newly "
            "created websites and social media. The reports argued against EU enlargement "
            "in Russia's sphere of influence. Infrastructure analysis linked the hosting "
            "to the same ASN used in prior Secondary Infektion operations."
        ),
        "campaign_id": "secondary_infektion",
        "reported_actor": "Secondary Infektion",
        "target_countries": ["EU", "UA"],
        "target_sectors": ["politics"],
        "first_seen": "2020-03-01",
        "last_seen": "2021-01-01",
        "ttps": ["T0008", "T0023"],
        "channels": ["twitter", "web"],
        "evidence_sources": [
            "https://eu.disinfo.eu/secondary-infektion-think-tanks/",
        ],
    },
    {
        "event_id": "SI-004",
        "title": "Secondary Infektion amplifies COVID-19 vaccine disinformation",
        "description": (
            "The operation pivoted to health disinformation, seeding false claims about "
            "Western COVID-19 vaccine side effects on Eastern European forums. The persona "
            "network reused previously observed account-creation scripts and shared hosting "
            "infrastructure with earlier Secondary Infektion clusters."
        ),
        "campaign_id": "secondary_infektion",
        "reported_actor": "Secondary Infektion",
        "target_countries": ["PL", "CZ", "SK", "HU"],
        "target_sectors": ["health"],
        "first_seen": "2021-01-01",
        "last_seen": "2021-12-31",
        "ttps": ["T0022", "T0023"],
        "channels": ["facebook", "web", "reddit"],
        "evidence_sources": [
            "https://eu.disinfo.eu/secondary-infektion-covid/",
        ],
    },

    # ── Unrelated events (5) — no campaign ──────────────────────────────
    {
        "event_id": "UR-001",
        "title": "Local climate disinformation on regional news site",
        "description": (
            "A regional news outlet published unsupported claims that climate change data "
            "was fabricated by international organisations. No attributable state actor; "
            "the content appeared to be financially motivated clickbait. No cyber indicators "
            "linked to known influence operation infrastructure."
        ),
        "campaign_id": None,
        "reported_actor": None,
        "target_countries": ["US"],
        "target_sectors": ["environment"],
        "first_seen": "2021-05-01",
        "last_seen": "2021-05-15",
        "ttps": ["T0022"],
        "channels": ["web"],
        "evidence_sources": [],
    },
    {
        "event_id": "UR-002",
        "title": "Domestic election rumour spread via WhatsApp",
        "description": (
            "Unverified electoral fraud claims circulated via WhatsApp groups in a Latin "
            "American country ahead of municipal elections. The origin appeared to be domestic "
            "political actors rather than foreign state-backed operations. No technical "
            "indicators connecting to foreign influence infrastructure were identified."
        ),
        "campaign_id": None,
        "reported_actor": None,
        "target_countries": ["BR"],
        "target_sectors": ["elections"],
        "first_seen": "2022-09-20",
        "last_seen": "2022-10-05",
        "ttps": ["T0049"],
        "channels": ["whatsapp"],
        "evidence_sources": [],
    },
    {
        "event_id": "UR-003",
        "title": "Anti-vaccine conspiracy on domestic social media",
        "description": (
            "A viral anti-vaccine conspiracy post alleged microchip implantation through "
            "injections. The content was created by a domestic account with no ties to "
            "foreign influence infrastructure. Rapid organic sharing led to significant reach "
            "without coordinated inauthentic amplification."
        ),
        "campaign_id": None,
        "reported_actor": None,
        "target_countries": ["GB"],
        "target_sectors": ["health"],
        "first_seen": "2021-07-01",
        "last_seen": "2021-07-10",
        "ttps": ["T0022"],
        "channels": ["facebook", "twitter"],
        "evidence_sources": [],
    },
    {
        "event_id": "UR-004",
        "title": "Financial fraud misinformation targeting elderly users",
        "description": (
            "Financially motivated actors spread false claims about a bank bailout to "
            "provoke panic selling. The operation used inauthentic social media accounts "
            "but with criminal rather than political motivation. No state nexus identified."
        ),
        "campaign_id": None,
        "reported_actor": None,
        "target_countries": ["US"],
        "target_sectors": ["finance"],
        "first_seen": "2023-03-01",
        "last_seen": "2023-03-05",
        "ttps": ["T0022"],
        "channels": ["twitter"],
        "evidence_sources": [],
    },
    {
        "event_id": "UR-005",
        "title": "Astroturfed pro-domestic-policy campaign",
        "description": (
            "A domestic political campaign used coordinated inauthentic accounts to "
            "simulate grassroots support for a controversial infrastructure bill. "
            "Analysis found bot-like posting patterns but no foreign infrastructure "
            "or state-backed actor attribution."
        ),
        "campaign_id": None,
        "reported_actor": None,
        "target_countries": ["AU"],
        "target_sectors": ["politics"],
        "first_seen": "2022-04-01",
        "last_seen": "2022-05-15",
        "ttps": ["T0022", "T0023"],
        "channels": ["twitter", "facebook"],
        "evidence_sources": [],
    },
]


def make_fixtures(out_dir: Path | None = None) -> list:
    from fimicyber.schema import Event
    from datetime import date as _date

    events: list[Event] = []
    for d in _EVENTS:
        def _parse_date(s: str | None):
            return _date.fromisoformat(s) if s else None

        ev = Event(
            event_id=d["event_id"],
            title=d["title"],
            description=d["description"],
            campaign_id=d.get("campaign_id"),
            campaign_id_source="fixture" if d.get("campaign_id") else "none",
            reported_actor=d.get("reported_actor"),
            target_countries=d.get("target_countries", []),
            target_sectors=d.get("target_sectors", []),
            first_seen=_parse_date(d.get("first_seen")),
            last_seen=_parse_date(d.get("last_seen")),
            ttps=d.get("ttps", []),
            channels=d.get("channels", []),
            evidence_sources=d.get("evidence_sources", []),
            source_dataset="fixture",
        )
        events.append(ev)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "events.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(ev.model_dump_json() + "\n")
        print(f"Fixtures written → {out_path} ({len(events)} events)")

    return events


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "interim")
    args = parser.parse_args()
    make_fixtures(args.out)
