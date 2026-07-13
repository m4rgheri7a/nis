"""Leakage and guardrail contracts for the evidence-structuring conditions.

These tests protect the claim the paper rests on: that the extraction conditions
derive their evidence from case text rather than inheriting curated gold fields.
"""
from __future__ import annotations

import pytest

from fimicyber.config import load_config
from fimicyber.eval.condition_benchmark import CONDITIONS, summarise_extraction
from fimicyber.llm.dossier import (
    AnnexRow,
    build_case_dossier,
    build_dossiers,
    build_label_scrubber,
)
from fimicyber.llm.evidence import (
    EvidenceCompiler,
    StructuredEvidence,
    apply_structured_evidence,
)
from fimicyber.loaders.external import load_multiactor_benchmark
from fimicyber.schema import IOC, Event


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def benchmark_events(cfg):
    events = load_multiactor_benchmark(cfg)
    if not events:
        pytest.skip("multi-campaign benchmark data is unavailable")
    return events


# ── Leakage control ────────────────────────────────────────────────────────


def test_dossiers_never_name_an_actor_or_campaign(benchmark_events, cfg):
    scrubber = build_label_scrubber(benchmark_events, cfg)
    dossiers = build_dossiers(benchmark_events, cfg)

    offenders = {
        event_id: found
        for event_id, text in dossiers.items()
        if (found := scrubber.find(text))
    }
    assert offenders == {}


def test_dossiers_never_render_curated_gold_labels(benchmark_events, cfg):
    """TTPs, channels and sectors are the answer key — they must not be in the input."""
    dossiers = build_dossiers(benchmark_events, cfg)

    for event in benchmark_events:
        text = dossiers[event.event_id]
        # The gold TTP list is rendered nowhere as a list; individual TTP words
        # may of course appear inside the prose the extractor is meant to read.
        assert "ttps" not in text.lower()
        assert "target_sectors" not in text.lower()
        assert event.campaign_id not in text
        assert "Campaign:" not in text
        assert "Actor:" not in text


def test_scrubber_redacts_a_label_planted_in_text(benchmark_events, cfg):
    scrubber = build_label_scrubber(benchmark_events, cfg)

    scrubbed = scrubber.scrub("The UNC1151 operators reused Doppelganger infrastructure.")

    assert "UNC1151" not in scrubbed
    assert "Doppelganger" not in scrubbed
    assert "infrastructure" in scrubbed


def test_scrubber_keeps_ordinary_vocabulary(benchmark_events, cfg):
    """Redaction must not shred the narrative the extractor has to read."""
    scrubber = build_label_scrubber(benchmark_events, cfg)

    text = "Internet research by a French agency described a storm of fake accounts."
    assert scrubber.scrub(text) == text


def test_dossier_annex_prints_indicators_defanged(benchmark_events, cfg):
    dossiers = build_dossiers(benchmark_events, cfg)

    text = dossiers["gw-test-2021-polskie-radio"]
    assert "88[.]99[.]132[.]118" in text
    assert "88.99.132.118" not in text


def test_annex_wording_does_not_flip_indicators_into_citations(benchmark_events, cfg):
    """The dossier must not sabotage the IOC classifier it feeds.

    ``ioc.classify`` weighs operational keywords against source keywords
    ("report", "source", "reference") near an observable. An annex header
    reading "indicators listed in the published report" tipped that balance and
    had real C2 addresses filed as EvidenceSourceURL and dropped. Extraction
    should now recover every gold indicator the regex layer can see at all.
    """
    from fimicyber.ioc.classify import classify_ioc
    from fimicyber.ioc.extract import _extract_raw_iocs

    dossiers = build_dossiers(benchmark_events, cfg)
    missed: list[tuple[str, str]] = []
    for event in benchmark_events:
        gold = {ioc.value.casefold() for ioc in event.iocs}
        if not gold:
            continue
        found = {
            value.casefold()
            for value, ioc_type, context in _extract_raw_iocs(dossiers[event.event_id])
            if classify_ioc(value, ioc_type, context, event.evidence_sources)[0]
            == "OperationalIOC"
        }
        missed += [(event.event_id, value) for value in gold - found]

    # The account handle matches no network pattern, so the regex layer cannot
    # see it by construction. Everything else must survive classification.
    assert missed == [("db-ref-2022-intrusion-truth", "intrusion_trutl")]


def test_annex_can_be_dropped_to_measure_ioc_sparsity(benchmark_events, cfg):
    with_annex = build_dossiers(benchmark_events, cfg, include_annex=True)
    without = build_dossiers(benchmark_events, cfg, include_annex=False)

    assert "Technical annex" in with_annex["gw-test-2021-polskie-radio"]
    assert "Technical annex" not in without["gw-test-2021-polskie-radio"]


# ── apply_structured_evidence must not fall back to gold ───────────────────


def test_empty_extraction_does_not_restore_curated_fields():
    """The bug this guards: `item.ttps or event.ttps` silently reinstated gold."""
    event = Event(
        event_id="case-1",
        title="Case",
        description="Some text.",
        ttps=["spoofed email"],
        channels=["Email"],
        target_sectors=["Government"],
        target_countries=["Poland"],
        iocs=[IOC(value="evil.example", ioc_type="domain",
                  category="OperationalIOC", confidence=0.9)],
    )
    empty = StructuredEvidence(event_id="case-1", backend="rules", core_narrative="")

    applied = apply_structured_evidence([event], [empty], replace=True)[0]

    assert applied.ttps == []
    assert applied.channels == []
    assert applied.target_sectors == []
    assert applied.target_countries == []
    assert applied.iocs == []


def test_apply_keeps_raw_description_as_narrative_by_default():
    """`description` is raw input, not a gold label; truncating it would not be a fix."""
    event = Event(event_id="case-2", title="Case",
                  description="First sentence. Second sentence.")
    evidence = StructuredEvidence(
        event_id="case-2", backend="rules", core_narrative="First sentence."
    )

    kept = apply_structured_evidence([event], [evidence], replace=True)[0]
    replaced = apply_structured_evidence(
        [event], [evidence], replace=True, replace_narrative=True
    )[0]

    assert kept.description == "First sentence. Second sentence."
    assert replaced.description == "First sentence."


def test_apply_promotes_extracted_iocs_into_the_graph_schema():
    event = Event(event_id="case-3", title="Case", description="Some text.")
    evidence = StructuredEvidence(
        event_id="case-3",
        backend="rules",
        core_narrative="Some text.",
        ioc_records=[
            IOC(value="1.2.3.4", ioc_type="ipv4", category="OperationalIOC",
                confidence=0.7).model_dump(mode="json")
        ],
    )

    applied = apply_structured_evidence([event], [evidence], replace=True)[0]

    assert [ioc.value for ioc in applied.iocs] == ["1.2.3.4"]
    assert applied.iocs[0].ioc_type == "ipv4"
    assert applied.llm_extracted is True


# ── Guardrails ─────────────────────────────────────────────────────────────


def test_defanged_annex_ioc_is_accepted_when_the_model_refangs_it():
    """The old substring check compared against the defanged text and lost real IOCs."""
    event = Event(event_id="case-4", title="Lure", description="A lure document.")
    dossier = (
        "CASE REPORT\nTitle: Lure\n\nSummary:\nA lure document.\n\n"
        "Technical annex - indicators listed in the published report (defanged):\n"
        "- 88[.]99[.]132[.]118 (IPv4 address) - C2 server.\n"
    )
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event, dossier)

    merged = compiler._merge_guarded(
        event, rules, {"ioc_candidates": ["88.99.132.118"]}, dossier
    )

    assert "88.99.132.118" in merged.ioc_candidates
    assert merged.hallucinated_iocs == []


def test_guardrail_keeps_an_annex_indicator_the_regex_pass_missed():
    """The guard is visibility, not regex agreement.

    An account handle matches no network pattern, so the regex extractor cannot
    see it. Discarding the model's reading of it — as an earlier merge did by
    intersecting with the regex output — threw away the LLM's only real IOC
    contribution and made the guarded condition identical to rules-only.
    """
    event = Event(event_id="case-9", title="Impersonator",
                  description="An impersonator account was documented.")
    dossier = (
        "CASE REPORT\nTitle: Impersonator\n\nSummary:\nAn impersonator account "
        "was documented.\n\n"
        "Technical annex - indicators listed in the published report (defanged):\n"
        "- intrusion_trutl (account handle) - Sample impersonator account.\n"
    )
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event, dossier)
    assert "intrusion_trutl" not in rules.ioc_candidates

    merged = compiler._merge_guarded(
        event, rules, {"ioc_candidates": ["intrusion_trutl"]}, dossier
    )

    assert "intrusion_trutl" in merged.ioc_candidates
    record = next(r for r in merged.ioc_records if r["value"] == "intrusion_trutl")
    assert record["ioc_type"] == "account"
    assert record["category"] == "OperationalIOC"


def test_guardrail_drops_a_visible_phrase_that_is_not_an_indicator():
    event = Event(event_id="case-10", title="Case",
                  description="Command-and-control infrastructure served RADIOSTAR payloads.")
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event)

    merged = compiler._merge_guarded(
        event,
        rules,
        {"ioc_candidates": ["command-and-control", "RADIOSTAR", "social media"]},
    )

    assert merged.ioc_candidates == []
    # Present in the text, so not hallucinated — just not an observable.
    assert "RADIOSTAR" not in merged.hallucinated_iocs


def test_guardrail_rejects_a_filename_that_looks_like_a_domain():
    event = Event(event_id="case-11", title="Lure",
                  description="The attachment News.03.doc carried the payload.")
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event)

    merged = compiler._merge_guarded(event, rules, {"ioc_candidates": ["News.03.doc"]})

    assert merged.ioc_candidates == []


def test_guardrail_rejects_and_records_a_hallucinated_ioc():
    event = Event(event_id="case-5", title="Case",
                  description="A fabricated report was amplified by social media accounts.")
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event)

    merged = compiler._merge_guarded(
        event, rules, {"ioc_candidates": ["plausible-but-absent.example", "9.9.9.9"]}
    )

    assert merged.ioc_candidates == []
    assert "plausible-but-absent.example" in merged.hallucinated_iocs


def test_unguarded_condition_lets_the_hallucination_through():
    """llm_only is the counterfactual that shows what the guardrails are catching."""
    event = Event(event_id="case-6", title="Case", description="A fabricated report.")
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event)

    unguarded = compiler._compile_unguarded(
        event,
        rules,
        {"ioc_candidates": ["invented.example"], "ttp_candidates": ["made up technique"]},
        None,
    )

    assert "invented.example" in unguarded.ioc_candidates
    assert "invented.example" in unguarded.hallucinated_iocs
    assert "made up technique" in unguarded.ttps


def test_rules_extraction_recovers_annex_indicators_from_the_dossier(benchmark_events, cfg):
    dossiers = build_dossiers(benchmark_events, cfg)
    event = next(e for e in benchmark_events if e.event_id == "dg-test-2023-media-clones-fr")

    evidence = EvidenceCompiler(mode="rules").compile_event(
        event, text=dossiers[event.event_id]
    )

    assert "lemonde.ltd" in evidence.ioc_candidates
    assert "France" in evidence.target_countries
    assert all(record["category"] == "OperationalIOC" for record in evidence.ioc_records)


def test_country_extraction_reads_demonyms():
    event = Event(
        event_id="case-7",
        title="Clones",
        description="Lookalike domains imitated major French newspapers and German outlets.",
    )

    evidence = EvidenceCompiler(mode="rules").compile_event(event)

    assert "France" in evidence.target_countries
    assert "Germany" in evidence.target_countries


# ── Condition wiring ───────────────────────────────────────────────────────


def test_condition_names_are_stable():
    assert CONDITIONS == ("curated_oracle", "rules_only", "llm_guarded", "llm_only")


def test_summarise_extraction_ignores_the_oracle_condition():
    """curated_oracle has no extraction step, so it must not appear in that table."""
    from fimicyber.eval.condition_benchmark import ConditionResult
    import pandas as pd

    oracle = ConditionResult(
        condition="curated_oracle",
        model="-",
        backend="curated",
        evaluation=pd.DataFrame(),
        predictions=pd.DataFrame(),
        class_metrics=pd.DataFrame(),
        error_analysis=pd.DataFrame(),
        hypotheses=pd.DataFrame(),
        extraction_per_event=None,
    )

    assert summarise_extraction([oracle]).empty


def test_dossier_renders_annex_context_but_not_source_urls():
    """Report URLs carry campaign names in their slugs, so they stay out of the text."""
    event = Event(
        event_id="case-8",
        title="Case",
        description="Some text.",
        evidence_sources=[
            "https://cloud.google.com/blog/topics/threat-intelligence/prc-dragonbridge-influence-elections/"
        ],
    )
    scrubber = build_label_scrubber([event], load_config())
    annex = [AnnexRow(value="evil.example", ioc_type="domain", context="Spoofed domain.")]

    text = build_case_dossier(event, annex, scrubber)

    assert "evil[.]example" in text
    assert "cloud.google.com" not in text
    assert "dragonbridge" not in text.lower()
