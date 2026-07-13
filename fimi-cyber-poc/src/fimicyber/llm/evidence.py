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

from fimicyber.schema import Event


_URL_RE = re.compile(r"(?:hxxps?|https?)://[^\s\"'<>)\]]+", re.I)
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.I)

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


@dataclass
class StructuredEvidence:
    event_id: str
    backend: str
    core_narrative: str
    target_sectors: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    ttps: list[str] = field(default_factory=list)
    ioc_candidates: list[str] = field(default_factory=list)
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
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_name
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.backend = "rules"
        self._tokenizer = None
        self._model = None
        if mode == "ollama":
            self.backend = f"ollama:{self.model_name}"
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

    def compile_event(self, event: Event) -> StructuredEvidence:
        if self.mode == "ollama":
            llm_obj = self._compile_with_ollama(event)
        else:
            llm_obj = self._compile_with_qwen(event) if self._model is not None else {}
        guarded = self._compile_with_rules(event)
        return self._merge_guarded(event, guarded, llm_obj)

    def _compile_with_ollama(self, event: Event) -> dict[str, Any]:
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
                {"role": "user", "content": _build_prompt(event)},
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

    def _compile_with_qwen(self, event: Event) -> dict[str, Any]:
        prompt = _build_prompt(event)
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

    def _compile_with_rules(self, event: Event) -> StructuredEvidence:
        text = f"{event.title}. {event.description}"
        lower = text.lower()
        ttps = _keyword_hits(lower, _TTP_KEYWORDS)
        channels = _keyword_hits(lower, _CHANNEL_KEYWORDS)
        targets = _keyword_hits(lower, _TARGET_KEYWORDS)
        iocs = sorted(set(_URL_RE.findall(text) + _DOMAIN_RE.findall(text)))

        ai_signal = "suspected" if any(
            k in lower for k in ("ai-generated", "deepfake", "synthetic", "fabricated video", "staged video")
        ) else "none"
        provenance = "missing" if any(
            k in lower for k in ("spoofed", "impersonating", "lookalike", "clone", "pseudo-media")
        ) else "unknown"
        stage = _infer_stage(lower, ttps, channels)
        uncertainty = []
        if not iocs:
            uncertainty.append("No explicit IOC candidate was visible in the text.")
        if ai_signal == "suspected":
            uncertainty.append("AI/synthetic signal is an investigative cue, not proof of manipulation.")
        return StructuredEvidence(
            event_id=event.event_id,
            backend=self.backend,
            core_narrative=_first_sentence(event.description),
            target_sectors=targets,
            channels=channels,
            ttps=ttps,
            ioc_candidates=iocs,
            ai_artifact_signal=ai_signal,
            provenance_signal=provenance,
            kill_chain_stage=stage,
            evidence_sentences=_evidence_sentences(event.description),
            uncertainty_notes=uncertainty,
        )

    def _merge_guarded(
        self,
        event: Event,
        rules: StructuredEvidence,
        llm_obj: dict[str, Any],
    ) -> StructuredEvidence:
        if not llm_obj:
            return rules

        text = f"{event.title}. {event.description}"
        lower = text.lower()

        def clean_list(name: str, fallback: list[str]) -> list[str]:
            raw = llm_obj.get(name, [])
            if isinstance(raw, str):
                raw = [raw]
            values = [str(v).strip() for v in raw if str(v).strip()]
            return sorted(set(values or fallback))

        # Map free-form LLM phrases back into the controlled evidence ontology.
        # This prevents invented labels from entering the graph as facts.
        ttps = sorted(set(rules.ttps) | set(_canonical_hits(clean_list("ttp_candidates", []), _TTP_KEYWORDS)))
        # Channels and target sectors describe directly observable routing and
        # affected entities. They are promoted only by deterministic textual
        # evidence; model-only guesses remain outside the evidence graph.
        channels = rules.channels
        targets = rules.target_sectors

        # IOC candidates must be visible technical observables, not semantic
        # phrases proposed by the model.
        raw_iocs = clean_list("ioc_candidates", [])
        visible_iocs = _URL_RE.findall(text) + _DOMAIN_RE.findall(text)
        llm_iocs = [
            value
            for value in raw_iocs
            if (_URL_RE.fullmatch(value) or _DOMAIN_RE.fullmatch(value))
            and value.lower() in lower
        ]
        iocs = sorted(set(rules.ioc_candidates) | set(visible_iocs) | set(llm_iocs))
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
            channels=channels,
            ttps=ttps,
            ioc_candidates=iocs,
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


def apply_structured_evidence(events: list[Event], evidence: list[StructuredEvidence]) -> list[Event]:
    by_id = {item.event_id: item for item in evidence}
    output: list[Event] = []
    for event in events:
        item = by_id.get(event.event_id)
        if item is None:
            output.append(event)
            continue
        updated = event.model_copy(deep=True)
        updated.description = item.core_narrative or updated.description
        updated.ttps = item.ttps or updated.ttps
        updated.channels = item.channels or updated.channels
        updated.target_sectors = item.target_sectors or updated.target_sectors
        updated.ai_artifact_signal = item.ai_artifact_signal
        updated.provenance_signal = item.provenance_signal
        updated.kill_chain_stage = item.kill_chain_stage
        updated.llm_extracted = True
        updated.evidence_sentences = item.evidence_sentences
        updated.uncertainty_notes = item.uncertainty_notes
        output.append(updated)
    return output


def evaluate_structuring(events: list[Event], evidence: list[StructuredEvidence]) -> pd.DataFrame:
    by_id = {item.event_id: item for item in evidence}
    rows: list[dict[str, Any]] = []
    for event in events:
        item = by_id[event.event_id]
        rows.append({
            "event_id": event.event_id,
            "backend": item.backend,
            "ttp_f1": _set_f1(event.ttps, item.ttps),
            "channel_f1": _set_f1(event.channels, item.channels),
            "target_f1": _set_f1(event.target_sectors, item.target_sectors),
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


def _build_prompt(event: Event) -> str:
    ttp_vocabulary = ", ".join(_TTP_KEYWORDS)
    channel_vocabulary = ", ".join(_CHANNEL_KEYWORDS)
    target_vocabulary = ", ".join(_TARGET_KEYWORDS)
    return f"""Extract evidence from this FIMI case. Return JSON only with keys:
core_narrative, target_sectors, channels, ttp_candidates, ioc_candidates,
ai_artifact_signal, provenance_signal, kill_chain_stage, evidence_sentences,
uncertainty.

Use only these exact controlled labels for target_sectors:
{target_vocabulary}
Use only these exact controlled labels for channels:
{channel_vocabulary}
Use only these exact controlled labels for ttp_candidates:
{ttp_vocabulary}
Include a label only when the text supplies direct evidence. Evidence sentences
must be verbatim sentences from the input. IOC candidates must appear verbatim
in the input. Return uncertainty as a JSON list of concrete limitations.

Allowed kill_chain_stage values: preparation, content_creation, seeding,
amplification, exploitation, unknown.
Allowed ai_artifact_signal values: none, suspected, confirmed, unknown.
Do not infer a state, organisation, or person attribution.

Title: {event.title}
Text: {event.description}
"""


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


def _set_f1(gold: list[str], pred: list[str]) -> float:
    gold_set = {v.lower() for v in gold}
    pred_set = {v.lower() for v in pred}
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    tp = len(gold_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(gold_set)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0
