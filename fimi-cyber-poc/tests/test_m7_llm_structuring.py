from fimicyber.llm.evidence import EvidenceCompiler
from fimicyber.schema import Event


def test_rules_compiler_extracts_guarded_evidence():
    event = Event(
        event_id="case-1",
        title="Spoofed ministry notice",
        description=(
            "A website impersonating a foreign ministry published a fabricated "
            "announcement. Coordinated social media accounts amplified "
            "https://example.com/story to public audiences."
        ),
        target_sectors=["Government", "Public"],
    )

    evidence = EvidenceCompiler(mode="rules").compile_event(event)

    assert evidence.kill_chain_stage in {"preparation", "amplification"}
    assert "Government" in evidence.target_sectors
    assert "Social media" in evidence.channels
    assert "https://example.com/story" in evidence.ioc_candidates
    assert evidence.ai_artifact_signal == "none"
    assert evidence.evidence_sentences


def test_llm_merge_does_not_accept_semantic_iocs():
    event = Event(
        event_id="case-2",
        title="Fabricated report",
        description="A fabricated report was amplified by social media accounts.",
    )
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event)
    merged = compiler._merge_guarded(
        event,
        rules,
        {
            "ioc_candidates": ["fabricated report", "social media"],
            "ai_artifact_signal": "confirmed",
            "ttp_candidates": ["fabricated article"],
        },
    )

    assert merged.ioc_candidates == []
    assert merged.ai_artifact_signal == "none"
    assert "fabricated article" in merged.ttps


def test_llm_merge_rejects_hallucinated_well_formed_ioc():
    event = Event(
        event_id="case-3",
        title="Fabricated report",
        description="A fabricated report was amplified by social media accounts.",
    )
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event)
    merged = compiler._merge_guarded(
        event,
        rules,
        {"ioc_candidates": ["plausible-but-absent.example"]},
    )

    assert merged.ioc_candidates == []


def test_llm_merge_requires_direct_target_and_evidence_sentence_support():
    event = Event(
        event_id="case-4",
        title="Fabricated report",
        description="A fabricated report was amplified by social media accounts.",
    )
    compiler = EvidenceCompiler(mode="rules")
    rules = compiler._compile_with_rules(event)
    merged = compiler._merge_guarded(
        event,
        rules,
        {
            "target_sectors": ["Government"],
            "evidence_sentences": ["A government was targeted."],
        },
    )

    assert "Government" not in merged.target_sectors
    assert "A government was targeted." not in merged.evidence_sentences
