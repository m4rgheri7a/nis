"""Qwen3-assisted evidence structuring with deterministic guardrails.

The LLM is used only to convert unstructured case text into evidence
candidates. Final ranking remains graph/scoring based.
"""
from __future__ import annotations

import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from fimicyber.ioc.extract import extract_iocs_from_text, refang
from fimicyber.schema import IOC, Event


_URL_RE = re.compile(r"(?:hxxps?|https?)://[^\s\"'<>)\]]+", re.I)
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.I)
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{32,64}\b")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")

_TTP_KEYWORDS: dict[str, tuple[str, ...]] = {
    "DDE": ("dde", "dynamic data exchange"),
    "PowerShell": ("powershell",),
    "RADIOSTAR": (),
    "URL redirection": ("url redirection", "redirected"),
    "VBS backdoor": ("vbs backdoor",),
    "VIDEOKILLER": (),
    "XOR encoded C2": ("xor encoded", "xor-encoded"),
    "actors": (),
    "ad hominem attack": ("ad hominem", "personal attacks"),
    "article plagiarism": ("article plagiarism", "plagiarized article"),
    "batch account creation": ("batch account", "accounts were created"),
    "bulk domain creation": ("bulk domain", "domains were registered"),
    "commercial PR distribution": ("commercial pr", "press-release distribution"),
    "content alteration": ("altered content", "content alteration"),
    "coordinated accounts": ("coordinated accounts",),
    "media website impersonation": ("news outlet", "newspaper", "media", "clone", "imitat"),
    "government website impersonation": ("government website", "ministry", "official", "spoofed government"),
    "coordinated amplification": ("amplified", "coordinated", "relayed", "distribution"),
    "coordinated dissemination": ("coordinated dissemination", "distributed together"),
    "coordinated inauthentic behavior": ("coordinated inauthentic", "inauthentic behavior"),
    "covert news website": ("covert news", "covert website"),
    "cross-domain amplification": ("cross-domain",),
    "cross-platform amplification": ("cross-platform", "multiple platforms"),
    "fabricated article": ("fabricated article", "false article", "article claimed", "stories claiming"),
    "fabricated announcement": ("fabricated announcement", "false announcement"),
    "fabricated cyber attribution": ("fabricated cyber attribution", "falsely attributed"),
    "fabricated statement": ("fabricated statement", "false statement"),
    "fabricated video": ("fabricated video", "staged video", "clip", "video"),
    "false location cues": ("false location", "location cues"),
    "staged testimony": ("testimony", "witness", "whistleblower"),
    "disposable account": ("disposable account", "ephemeral account", "newly created account"),
    "disposable persona": ("disposable persona", "fake persona"),
    "editorial concealment": ("editorial concealment", "concealed sponsorship"),
    "election interference narrative": ("election interference",),
    "election targeting": ("election targeting", "targeted the election"),
    "ephemeral account": ("ephemeral account", "single-use account"),
    "hashtag amplification": ("hashtag",),
    "inauthentic local news sites": ("inauthentic local news", "local news sites"),
    "localized content copying": ("localized copies", "localized content"),
    "malicious document": ("malicious document", "malicious dde document"),
    "malicious presentation": ("malicious presentation", "malicious powerpoint"),
    "multi-stage laundering": ("multi-stage", "multiple stages"),
    "narrative laundering": ("narrative laundering", "laundered narrative"),
    "paid advertisements": ("paid advertisements", "paid ads"),
    "paid promotion": ("paid promotion", "promoted posts"),
    "persona impersonation": ("persona impersonation", "impersonated persona"),
    "post plagiarism": ("post plagiarism", "plagiarized posts"),
    "profile image reuse": ("profile image reuse", "reused profile"),
    "proxy media amplification": ("proxy media",),
    "pseudo-media website": ("pseudo-media", "fake media website"),
    "single-use accounts": ("single-use account",),
    "social amplification": ("social amplification", "social media amplified"),
    "social media account compromise": ("compromised social", "social-media accounts"),
    "spear phishing": ("spear phishing", "spear-phishing"),
    "spoofed email": ("spoofed email", "spoofed e-mail"),
    "spoofed government website": ("spoofed government",),
    "staged video": ("staged video",),
    "staged whistleblower": ("staged whistleblower", "fake whistleblower"),
    "staged witness": ("staged witness", "fake witness"),
    "system discovery": ("system discovery",),
    "tracking pixel": ("tracking pixel",),
    "website compromise": ("compromised website", "compromised news"),
    "typosquatting": ("lookalike domain", "typosquat", "lookalike"),
    "web shell": ("web shell", "webshell"),
    "content syndication": ("syndication", "republished", "press release"),
    "multilingual dissemination": ("multilingual", "several languages", "english and chinese"),
}

_CHANNEL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Blogs": ("blog", "blogs"),
    "Compromised website": ("compromised website", "compromised news"),
    "Social media": ("social media", "social-media", "accounts"),
    "Facebook": ("facebook",),
    "Instagram": ("instagram",),
    "Twitter": ("twitter", "x.com"),
    "Telegram": ("telegram",),
    "Email": ("email", "journalists", "recipients"),
    "Video platform": ("video", "clip"),
    "Impersonation website": ("impersonating", "lookalike", "spoofed", "clone"),
    "Covert news website": ("covert news", "pseudo-media", "proxy media"),
    "Forums": ("forum", "forums"),
    "Inauthentic news website": ("inauthentic news", "local news sites"),
    "Newswire": ("newswire", "press-release", "press release"),
    "Online advertising": ("advertisements", "advertising", "paid promotion"),
}

_TARGET_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Civil society": ("civil society", "ngo", "activists"),
    "Government": ("government", "ministry", "official", "parliamentary", "minister"),
    "Elections": ("election", "voting", "ballots", "candidate"),
    "Media": ("media", "news", "journalists", "newspaper"),
    "Public": ("citizens", "public", "audiences", "readers"),
    "Defence": ("military", "soldiers", "armed forces", "defence"),
    "Cybersecurity": ("cyber", "cybersecurity", "apt"),
    "Diaspora": ("diaspora", "overseas communities"),
    "Energy": ("nuclear", "energy", "radioactive"),
}


# Target countries are part of the ``target`` evidence family (weight 0.10 in
# the attribution score). Without an extractor they would stay curated-oracle in
# every condition, so the extraction conditions would silently keep a gold
# feature. The gazetteer covers the vocabulary used by the benchmark's gold
# labels plus the demonyms the reports actually use.
_COUNTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Lithuania": ("lithuania", "lithuanian", "vilnius"),
    "Poland": ("poland", "polish", "warsaw"),
    "Ukraine": ("ukraine", "ukrainian", "kyiv", "kiev"),
    "United States": ("united states", "u.s.", "american", "washington"),
    "Germany": ("germany", "german", "berlin", "bundestag"),
    "France": ("france", "french", "paris"),
    "Italy": ("italy", "italian", "rome"),
    "United Kingdom": ("united kingdom", "british", "britain", "london"),
    "European Union": ("european union", "brussels", "eu institutions", "european commission"),
    "China": ("china", "chinese", "prc", "beijing"),
    "Taiwan": ("taiwan", "taiwanese", "taipei"),
    "Eastern Europe": ("eastern europe",),
    "Global": ("worldwide", "globally", "multiple countries", "several countries"),
}


@dataclass
class StructuredEvidence:
    event_id: str
    backend: str
    core_narrative: str
    target_sectors: list[str] = field(default_factory=list)
    target_countries: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    ttps: list[str] = field(default_factory=list)
    ioc_candidates: list[str] = field(default_factory=list)
    ioc_records: list[dict[str, Any]] = field(default_factory=list)
    hallucinated_iocs: list[str] = field(default_factory=list)
    ai_artifact_signal: str = "unknown"
    provenance_signal: str = "unknown"
    kill_chain_stage: str = "unknown"
    evidence_sentences: list[str] = field(default_factory=list)
    uncertainty_notes: list[str] = field(default_factory=list)


class EvidenceCompiler:
    """Compile unstructured FIMI text into guarded evidence objects."""

    def __init__(
        self,
        model_name: str = "qwen3:8b",
        mode: str = "rules",
        max_new_tokens: int = 512,
    ) -> None:
        self.model_name = model_name
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.backend = "rules"
        self._tokenizer = None
        self._model = None
        if mode in {"ollama", "llm_only"}:
            self.backend = f"ollama:{self.model_name}"
            if mode == "llm_only":
                self.backend += "(unguarded)"
        elif mode in {"auto", "qwen3"}:
            self._try_load_qwen()

    def _try_load_qwen(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            self.backend = f"qwen3:{self.model_name}"
        except Exception as exc:
            if self.mode == "qwen3":
                raise RuntimeError(f"Qwen3 load failed: {exc}") from exc
            self.backend = f"rules(qwen_unavailable:{type(exc).__name__})"

    def compile_event(self, event: Event, text: str | None = None) -> StructuredEvidence:
        """Structure one case.

        ``text`` is the document the extractor is allowed to read. When omitted
        it falls back to the event's own title and description, which keeps the
        original behaviour for callers that have no dossier.
        """
        source = _source_text(event, text)
        if self.mode in {"ollama", "llm_only"}:
            llm_obj = self._compile_with_ollama(event, source)
        else:
            llm_obj = (
                self._compile_with_qwen(event, source) if self._model is not None else {}
            )
        rules = self._compile_with_rules(event, source)
        if self.mode == "llm_only":
            return self._compile_unguarded(event, rules, llm_obj, source)
        return self._merge_guarded(event, rules, llm_obj, source)

    def _compile_with_ollama(self, event: Event, text: str | None = None) -> dict[str, Any]:
        payload = {
            "model": self.model_name,
            "stream": False,
            "format": "json",
            "think": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You extract evidence only. Return compact JSON. Do not attribute actors.",
                },
                {"role": "user", "content": _build_prompt(event, text)},
            ],
            "options": {
                "temperature": 0,
                "num_predict": self.max_new_tokens,
                "seed": 42,
            },
        }
        request = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Ollama inference failed for {event.event_id}: {exc}") from exc
        return _parse_json_object(str(body.get("message", {}).get("content", "")))

    def _compile_with_qwen(self, event: Event, text: str | None = None) -> dict[str, Any]:
        prompt = _build_prompt(event, text)
        tokenizer = self._tokenizer
        model = self._model
        if tokenizer is None or model is None:
            return {}
        messages = [
            {"role": "system", "content": "You extract evidence only. Return compact JSON. Do not attribute actors."},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )
        return _parse_json_object(response)

    def _compile_with_rules(
        self, event: Event, text: str | None = None
    ) -> StructuredEvidence:
        source = _source_text(event, text)
        lower = source.lower()
        ttps = _keyword_hits(lower, _TTP_KEYWORDS)
        channels = _keyword_hits(lower, _CHANNEL_KEYWORDS)
        targets = _keyword_hits(lower, _TARGET_KEYWORDS)
        countries = _keyword_hits(lower, _COUNTRY_KEYWORDS)

        # Typed IOC objects go into the evidence graph, so they run through the
        # same refang → classify → confidence path as the main IOC pipeline
        # rather than being bare strings.
        ioc_objects = [
            ioc
            for ioc in extract_iocs_from_text(source, event)
            if ioc.category == "OperationalIOC"
        ]
        ioc_records = [ioc.model_dump(mode="json") for ioc in ioc_objects]
        ioc_values = sorted({ioc.value for ioc in ioc_objects})

        ai_signal = "suspected" if any(
            k in lower for k in ("ai-generated", "deepfake", "synthetic", "fabricated video", "staged video")
        ) else "none"
        provenance = "missing" if any(
            k in lower for k in ("spoofed", "impersonating", "lookalike", "clone", "pseudo-media")
        ) else "unknown"
        stage = _infer_stage(lower, ttps, channels)
        uncertainty = []
        if not ioc_values:
            uncertainty.append("No explicit IOC candidate was visible in the text.")
        if ai_signal == "suspected":
            uncertainty.append("AI/synthetic signal is an investigative cue, not proof of manipulation.")
        return StructuredEvidence(
            event_id=event.event_id,
            backend=self.backend,
            core_narrative=_first_sentence(_summary_of(source, event)),
            target_sectors=targets,
            target_countries=countries,
            channels=channels,
            ttps=ttps,
            ioc_candidates=ioc_values,
            ioc_records=ioc_records,
            ai_artifact_signal=ai_signal,
            provenance_signal=provenance,
            kill_chain_stage=stage,
            evidence_sentences=_evidence_sentences(_summary_of(source, event)),
            uncertainty_notes=uncertainty,
        )

    def _merge_guarded(
        self,
        event: Event,
        rules: StructuredEvidence,
        llm_obj: dict[str, Any],
        text: str | None = None,
    ) -> StructuredEvidence:
        if not llm_obj:
            return rules

        source = _source_text(event, text)
        lower = source.lower()
        # Public annexes print indicators defanged ("88[.]99[.]132[.]118"), while
        # a model asked for indicators answers with the live form. Checking
        # visibility against the raw text alone would therefore discard the very
        # IOCs the annex supplies and count them as hallucinations.
        refanged_lower = refang(source).lower()

        def clean_list(name: str, fallback: list[str]) -> list[str]:
            raw = llm_obj.get(name, [])
            if isinstance(raw, str):
                raw = [raw]
            values = [str(v).strip() for v in raw if str(v).strip()]
            return sorted(set(values or fallback))

        # Map free-form LLM phrases back into the controlled evidence ontology.
        # This prevents invented labels from entering the graph as facts.
        ttps = sorted(set(rules.ttps) | set(_canonical_hits(clean_list("ttp_candidates", []), _TTP_KEYWORDS)))
        # Channels, target sectors and target countries describe directly
        # observable routing and affected entities. They are promoted only by
        # deterministic textual evidence; model-only guesses stay outside the
        # evidence graph.
        channels = rules.channels
        targets = rules.target_sectors
        countries = rules.target_countries

        # Two independent gates, in this order:
        #   1. visibility — an indicator absent from the source is a hallucination,
        #      whatever shape it has;
        #   2. observability — a phrase that *is* in the source ("social media")
        #      is still not a technical indicator, so it is dropped silently.
        raw_iocs = clean_list("ioc_candidates", [])
        accepted_llm_iocs: list[str] = []
        hallucinated: list[str] = []
        for value in raw_iocs:
            if not _visible_in(value, lower, refanged_lower):
                hallucinated.append(value)
                continue
            if _is_wellformed_ioc(value):
                accepted_llm_iocs.append(refang(value).strip())

        # A surviving model value still goes through the same refang → classify →
        # confidence path as the regex extractor, so nothing untyped or unscored
        # reaches the evidence graph. The regex pass wins on duplicates because it
        # carries a context window it found itself.
        records: dict[str, dict[str, Any]] = {
            record["value"]: record for record in rules.ioc_records
        }
        refanged_source = refang(source)
        for value in accepted_llm_iocs:
            if value in records:
                continue
            record = _record_for_llm_ioc(value, refanged_source, event)
            if record is not None:
                records[record["value"]] = record
        ioc_records = list(records.values())
        confirmed = sorted(records)

        uncertainty = sorted(set(rules.uncertainty_notes) | set(clean_list("uncertainty", [])))
        ai_signal = rules.ai_artifact_signal
        if any(k in lower for k in ("deepfake", "ai-generated", "synthetic media")):
            ai_signal = "suspected"
        llm_evidence_sentences = [
            sentence
            for sentence in clean_list("evidence_sentences", [])
            if sentence.lower() in lower
        ]
        return StructuredEvidence(
            event_id=event.event_id,
            backend=self.backend,
            core_narrative=str(llm_obj.get("core_narrative") or rules.core_narrative),
            target_sectors=targets,
            target_countries=countries,
            channels=channels,
            ttps=ttps,
            ioc_candidates=confirmed,
            ioc_records=ioc_records,
            hallucinated_iocs=sorted(set(hallucinated)),
            ai_artifact_signal=ai_signal,
            provenance_signal=_safe_choice(
                str(llm_obj.get("provenance_signal", rules.provenance_signal)),
                {"present", "missing", "stripped", "inconsistent", "unknown"},
                rules.provenance_signal,
            ),
            kill_chain_stage=_safe_choice(
                str(llm_obj.get("kill_chain_stage", rules.kill_chain_stage)),
                {"preparation", "content_creation", "seeding", "amplification", "exploitation", "unknown"},
                rules.kill_chain_stage,
            ),
            evidence_sentences=llm_evidence_sentences or rules.evidence_sentences,
            uncertainty_notes=uncertainty,
        )

    def _compile_unguarded(
        self,
        event: Event,
        rules: StructuredEvidence,
        llm_obj: dict[str, Any],
        text: str | None = None,
    ) -> StructuredEvidence:
        """Take the model at its word — the baseline the guardrails exist to beat.

        Free-form labels enter the ontology unmapped and indicators are accepted
        without checking that they appear in the source, so this condition shows
        what reaches the evidence graph when nothing filters the model.
        """
        source = _source_text(event, text)
        refanged_lower = refang(source).lower()

        def raw_list(name: str) -> list[str]:
            raw = llm_obj.get(name, [])
            if isinstance(raw, str):
                raw = [raw]
            return sorted({str(v).strip() for v in raw if str(v).strip()})

        ioc_values = raw_list("ioc_candidates")
        hallucinated = [
            value
            for value in ioc_values
            if refang(value).lower() not in refanged_lower
        ]
        ioc_records = [
            IOC(
                value=refang(value),
                ioc_type=_guess_ioc_type(value),
                category="OperationalIOC",
                confidence=0.5,
                conf_components={"llm_only": 1.0},
                status="candidate",
                synthetic=False,
            ).model_dump(mode="json")
            for value in ioc_values
        ]
        return StructuredEvidence(
            event_id=event.event_id,
            backend=self.backend,
            core_narrative=str(llm_obj.get("core_narrative") or ""),
            target_sectors=raw_list("target_sectors"),
            target_countries=raw_list("target_countries"),
            channels=raw_list("channels"),
            ttps=raw_list("ttp_candidates"),
            ioc_candidates=[refang(value) for value in ioc_values],
            ioc_records=ioc_records,
            hallucinated_iocs=sorted(set(hallucinated)),
            ai_artifact_signal=str(llm_obj.get("ai_artifact_signal") or "unknown"),
            provenance_signal=str(llm_obj.get("provenance_signal") or "unknown"),
            kill_chain_stage=str(llm_obj.get("kill_chain_stage") or "unknown"),
            evidence_sentences=raw_list("evidence_sentences"),
            uncertainty_notes=raw_list("uncertainty"),
        )


def apply_structured_evidence(
    events: list[Event],
    evidence: list[StructuredEvidence],
    replace: bool = True,
    replace_narrative: bool = False,
) -> list[Event]:
    """Write structured evidence back onto the events that feed the graph.

    ``replace=True`` is what an honest extraction condition requires: when the
    extractor finds no TTP, the event must end up with no TTP. The previous
    ``item.ttps or event.ttps`` fallback quietly restored the curated gold list
    whenever extraction came up empty, which let an extraction condition inherit
    the oracle's features and score as if it had derived them.

    ``description`` is raw input rather than a curated label, so it is left alone
    by default. Overwriting it with a one-line ``core_narrative`` would discard
    most of the text the narrative embedding is computed from — an information
    loss unrelated to extraction quality. ``replace_narrative=True`` exists to
    measure that variant deliberately.
    """
    by_id = {item.event_id: item for item in evidence}
    output: list[Event] = []
    for event in events:
        item = by_id.get(event.event_id)
        if item is None:
            output.append(event)
            continue
        updated = event.model_copy(deep=True)

        if replace_narrative and item.core_narrative:
            updated.description = item.core_narrative

        if replace:
            updated.ttps = list(item.ttps)
            updated.channels = list(item.channels)
            updated.target_sectors = list(item.target_sectors)
            updated.target_countries = list(item.target_countries)
            updated.iocs = [IOC.model_validate(record) for record in item.ioc_records]
        else:
            updated.ttps = item.ttps or updated.ttps
            updated.channels = item.channels or updated.channels
            updated.target_sectors = item.target_sectors or updated.target_sectors
            updated.target_countries = item.target_countries or updated.target_countries

        updated.ai_artifact_signal = item.ai_artifact_signal
        updated.provenance_signal = item.provenance_signal
        updated.kill_chain_stage = item.kill_chain_stage
        updated.llm_extracted = True
        updated.evidence_sentences = item.evidence_sentences
        updated.uncertainty_notes = item.uncertainty_notes
        output.append(updated)
    return output


def evaluate_structuring(events: list[Event], evidence: list[StructuredEvidence]) -> pd.DataFrame:
    """Score structured evidence against the curated gold fields.

    ``events`` must carry the curated labels: pass the events as loaded, not the
    ones already rewritten by :func:`apply_structured_evidence`.
    """
    by_id = {item.event_id: item for item in evidence}
    rows: list[dict[str, Any]] = []
    for event in events:
        item = by_id[event.event_id]
        gold_iocs = {ioc.value.casefold() for ioc in event.iocs if not ioc.synthetic}
        found_iocs = {value.casefold() for value in item.ioc_candidates}
        precision, recall, f1 = _set_prf(gold_iocs, found_iocs)
        rows.append({
            "event_id": event.event_id,
            "backend": item.backend,
            "ttp_precision": _set_prf(set(event.ttps), set(item.ttps))[0],
            "ttp_recall": _set_prf(set(event.ttps), set(item.ttps))[1],
            "ttp_f1": _set_f1(event.ttps, item.ttps),
            "channel_precision": _set_prf(set(event.channels), set(item.channels))[0],
            "channel_recall": _set_prf(set(event.channels), set(item.channels))[1],
            "channel_f1": _set_f1(event.channels, item.channels),
            "target_precision": _set_prf(set(event.target_sectors), set(item.target_sectors))[0],
            "target_recall": _set_prf(set(event.target_sectors), set(item.target_sectors))[1],
            "target_f1": _set_f1(event.target_sectors, item.target_sectors),
            "country_f1": _set_f1(event.target_countries, item.target_countries),
            "gold_ioc_count": len(gold_iocs),
            "ioc_precision": precision,
            "ioc_recall": recall,
            "ioc_f1": f1,
            "hallucinated_ioc_count": len(item.hallucinated_iocs),
            "hallucinated_iocs": "|".join(item.hallucinated_iocs),
            "evidence_sentence_count": len(item.evidence_sentences),
            "ioc_candidate_count": len(item.ioc_candidates),
            "uncertainty_count": len(item.uncertainty_notes),
            "ai_artifact_signal": item.ai_artifact_signal,
            "provenance_signal": item.provenance_signal,
            "kill_chain_stage": item.kill_chain_stage,
        })
    return pd.DataFrame(rows)


def write_structuring_outputs(
    events: list[Event],
    evidence: list[StructuredEvidence],
    out_dir: Path,
    suffix: str = "",
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix_part = f"_{suffix.strip()}" if suffix.strip() else ""
    jsonl_path = out_dir / f"llm_structured_evidence{suffix_part}.jsonl"
    csv_path = out_dir / f"llm_structuring_evaluation{suffix_part}.csv"
    summary_path = out_dir / f"llm_structuring_summary{suffix_part}.md"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in evidence:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    eval_df = evaluate_structuring(events, evidence)
    eval_df.to_csv(csv_path, index=False)
    summary_path.write_text(_summary_markdown(eval_df, evidence), encoding="utf-8")
    return {"jsonl": jsonl_path, "csv": csv_path, "summary": summary_path}


def _summary_markdown(eval_df: pd.DataFrame, evidence: Iterable[StructuredEvidence]) -> str:
    backend = next(iter(evidence)).backend if evidence else "unknown"
    lines = [
        "# LLM Evidence Structuring 결과",
        "",
        f"- backend: `{backend}`",
        f"- events: {len(eval_df)}",
        f"- mean TTP F1: {eval_df['ttp_f1'].mean():.3f}",
        f"- mean channel F1: {eval_df['channel_f1'].mean():.3f}",
        f"- mean target F1: {eval_df['target_f1'].mean():.3f}",
        f"- events with evidence sentence: {(eval_df['evidence_sentence_count'] > 0).mean():.3f}",
        f"- events with IOC candidate: {(eval_df['ioc_candidate_count'] > 0).mean():.3f}",
        "",
        "LLM은 최종 귀속 판단자가 아니라 비정형 자료를 증거 객체로 바꾸는 전처리 모듈로 사용되었다.",
    ]
    return "\n".join(lines) + "\n"


def _build_prompt(event: Event, text: str | None = None) -> str:
    ttp_vocabulary = ", ".join(_TTP_KEYWORDS)
    channel_vocabulary = ", ".join(_CHANNEL_KEYWORDS)
    target_vocabulary = ", ".join(_TARGET_KEYWORDS)
    country_vocabulary = ", ".join(_COUNTRY_KEYWORDS)
    source = _source_text(event, text)
    return f"""Extract evidence from this FIMI case report. Return JSON only with keys:
core_narrative, target_sectors, target_countries, channels, ttp_candidates,
ioc_candidates, ai_artifact_signal, provenance_signal, kill_chain_stage,
evidence_sentences, uncertainty.

Use only these exact controlled labels for target_sectors:
{target_vocabulary}
Use only these exact controlled labels for target_countries:
{country_vocabulary}
Use only these exact controlled labels for channels:
{channel_vocabulary}
Use only these exact controlled labels for ttp_candidates:
{ttp_vocabulary}
Include a label only when the report supplies direct evidence. Evidence
sentences must be verbatim sentences from the report. IOC candidates must be
technical indicators that appear in the report: domains, URLs, IP addresses,
file hashes, e-mail addresses, or account handles. The technical annex prints
them defanged, so "example[.]com" means the domain example.com and
"1[.]2[.]3[.]4" means the address 1.2.3.4; return the restored form. Never
invent an indicator that is not printed in the report. Return uncertainty as a
JSON list of concrete limitations.

Allowed kill_chain_stage values: preparation, content_creation, seeding,
amplification, exploitation, unknown.
Allowed ai_artifact_signal values: none, suspected, confirmed, unknown.
Do not infer a state, organisation, campaign, or person attribution.

REPORT:
{source}
"""


_SUMMARY_RE = re.compile(r"^Summary:\s*\n(.*?)(?:\n\s*\n|\Z)", re.S | re.M)


def _source_text(event: Event, text: str | None = None) -> str:
    """The document an extractor is allowed to read."""
    if text and text.strip():
        return text
    return f"{event.title}. {event.description}"


def _summary_of(source: str, event: Event) -> str:
    """The narrative prose inside a dossier, excluding the technical annex."""
    match = _SUMMARY_RE.search(source)
    if match:
        return match.group(1).strip()
    return event.description or source


_FILE_EXT_RE = re.compile(
    r"\.(?:docx?|xlsx?|pptx?|pdf|zip|rar|exe|dll|js|vbs|ps1|txt|rtf|lnk)$", re.I
)


def _valid_domain(candidate: str) -> bool:
    """A registrable domain, not a filename that happens to contain dots."""
    if _FILE_EXT_RE.search(candidate):
        return False
    try:
        import tldextract

        extracted = tldextract.extract(candidate)
        return bool(extracted.domain and extracted.suffix)
    except Exception:
        return False


def _looks_like_handle(candidate: str) -> bool:
    """An opaque identifier such as an account name, not a phrase.

    Reports list account handles among their indicators, and they match no
    network pattern. Requiring an underscore or a digit keeps prose out:
    "command-and-control" and "RADIOSTAR" are rejected, "intrusion_trutl" is not.
    """
    if len(candidate) < 6 or candidate.isalpha() or _FILE_EXT_RE.search(candidate):
        return False
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._@-]*", candidate):
        return False
    return "_" in candidate or any(char.isdigit() for char in candidate)


def _is_wellformed_ioc(value: str) -> bool:
    """True when a string is a technical observable rather than a phrase."""
    candidate = refang(value).strip()
    if not candidate or any(char.isspace() for char in candidate):
        return False
    if _IPV4_RE.fullmatch(candidate) or _HASH_RE.fullmatch(candidate):
        return True
    if _URL_RE.fullmatch(candidate):
        return True
    if _DOMAIN_RE.fullmatch(candidate) and _valid_domain(candidate):
        return True
    return _looks_like_handle(candidate)


def _record_for_llm_ioc(
    value: str, refanged_source: str, event: Event
) -> dict[str, Any] | None:
    """Type, classify and score a model-proposed indicator the way the regex pass does."""
    from fimicyber.ioc.classify import classify_ioc
    from fimicyber.ioc.confidence import compute_confidence

    candidate = refang(value).strip()
    index = refanged_source.casefold().find(candidate.casefold())
    if index < 0:
        return None
    context = refanged_source[max(0, index - 120) : index + len(candidate) + 120]

    ioc_type = _guess_ioc_type(candidate)
    category, status = classify_ioc(
        value=candidate,
        ioc_type=ioc_type,
        context=context,
        evidence_sources=event.evidence_sources,
    )
    if category != "OperationalIOC":
        return None

    components, confidence = compute_confidence(
        ioc_type=ioc_type,
        category=category,
        context=context,
        source_label="llm_extraction",
        n_sources=1,
        event_first_seen=event.first_seen,
        event_last_seen=event.last_seen,
        ioc_first_seen=None,
        ioc_last_seen=None,
    )
    return IOC(
        value=candidate,
        ioc_type=ioc_type,
        category=category,
        confidence=max(0.0, min(1.0, confidence)),
        conf_components=components,
        sources=event.evidence_sources[:3],
        status=status,
        synthetic=False,
    ).model_dump(mode="json")


def _visible_in(value: str, lower: str, refanged_lower: str) -> bool:
    """True when the indicator is printed in the source, defanged or not."""
    candidate = value.strip().lower()
    return (
        candidate in lower
        or candidate in refanged_lower
        or refang(candidate) in refanged_lower
    )


def _guess_ioc_type(value: str) -> str:
    candidate = refang(value).strip()
    if _IPV4_RE.fullmatch(candidate):
        return "ipv4"
    if _URL_RE.fullmatch(candidate):
        return "url"
    if re.fullmatch(r"[0-9a-fA-F]{64}", candidate):
        return "hash_sha256"
    if re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
        return "hash_sha1"
    if re.fullmatch(r"[0-9a-fA-F]{32}", candidate):
        return "hash_md5"
    if "@" in candidate:
        return "email"
    if _DOMAIN_RE.fullmatch(candidate) and _valid_domain(candidate):
        return "domain"
    return "account"


def _parse_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _keyword_hits(text: str, mapping: dict[str, tuple[str, ...]]) -> list[str]:
    return sorted({label for label, needles in mapping.items() if any(n in text for n in needles)})


def _canonical_hits(values: list[str], mapping: dict[str, tuple[str, ...]]) -> list[str]:
    hits: set[str] = set()
    for value in values:
        value_l = value.lower()
        for label, needles in mapping.items():
            if label.lower() in value_l or any(needle in value_l for needle in needles):
                hits.add(label)
    return sorted(hits)


def _infer_stage(text: str, ttps: list[str], channels: list[str]) -> str:
    if any(k in text for k in ("domain", "website", "impersonating", "clone", "lookalike")):
        return "preparation"
    if any(k in " ".join(ttps).lower() for k in ("fabricated", "staged", "content")):
        return "content_creation"
    if any(k in text for k in ("seeded", "first appeared", "disposable")):
        return "seeding"
    if any(k in text for k in ("amplified", "promoted", "relayed", "advertising", "accounts")):
        return "amplification"
    if any(k in text for k in ("harm", "interference", "burden", "russophobia")):
        return "exploitation"
    return "unknown"


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts and parts[0] else text.strip()


def _evidence_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    return sentences[:3]


def _safe_choice(value: str, allowed: set[str], fallback: str) -> str:
    return value if value in allowed else fallback


def _set_prf(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    """Precision, recall, F1 over two label sets, matched case-insensitively."""
    gold_set = {v.casefold() for v in gold}
    pred_set = {v.casefold() for v in pred}
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set or not pred_set:
        return 0.0, 0.0, 0.0
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _set_f1(gold: list[str], pred: list[str]) -> float:
    return _set_prf(set(gold), set(pred))[2]
