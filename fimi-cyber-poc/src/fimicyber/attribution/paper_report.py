"""Write a conservative, paper-ready summary of attribution validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _pct(value: Any) -> str:
    return f"{100 * float(value):.2f}%"


def write_paper_validation_report(
    metrics: pd.DataFrame,
    calibration: pd.DataFrame,
    external_case: dict[str, Any],
    output_path: Path,
) -> None:
    validation = metrics[metrics["evaluation_scope"] == "validation"].iloc[0]
    external = metrics[metrics["evaluation_scope"] == "external_ghostwriter"].iloc[0]
    cal = calibration.iloc[0]
    comparison = external_case["condition_comparison"].copy()
    comparison = comparison.fillna("")
    decisions = external_case["hypotheses"]
    decisions = decisions[decisions["rank"] == 1][[
        "query_event_id", "candidate_actor", "assessment_confidence",
        "decision", "abstention_reason", "real_ioc_support",
        "supporting_event_ids",
    ]].copy()
    decisions = decisions.fillna("")
    decisions["supporting_event_ids"] = decisions["supporting_event_ids"].str.replace(
        "|", "<br>", regex=False
    )

    lines = [
        "# 행위자 귀속지원 및 Ghostwriter 외부사례 검증 결과",
        "",
        "## 1. 검증 설계",
        "",
        "행위자 라벨은 질의 사건의 입력 특징으로 사용하지 않고, 질의 시점보다 먼저 공개된 사건을 "
        "행위자별 참조 프로필로 묶는 데에만 사용하였다. 합성 IOC는 귀속지원 점수에서 제외하였다. "
        "행위자 별칭은 명시적 taxonomy로 정규화하되 국가, 생태계, 조직, 캠페인 수준은 서로 합치지 않았다.",
        "",
        "내부 자료는 사건 공개일 순으로 보정 구간 60%와 검증 구간 40%로 나누었다. "
        f"보정 구간 {int(cal['calibration_queries'])}건에서 temperature를 "
        f"{float(cal['baseline_temperature']):.3f}에서 {float(cal['fitted_temperature']):.3f}으로 조정했으며, "
        f"NLL은 {float(cal['baseline_calibration_NLL']):.3f}에서 "
        f"{float(cal['fitted_calibration_NLL']):.3f}으로 감소하였다.",
        "",
        "## 2. 내부 시간분리 검증",
        "",
        f"검증 구간은 {int(validation['queries_evaluable'])}건, 행위자 라벨 "
        f"{int(validation['n_actor_labels'])}종으로 구성되었다. Top-1 정확도는 "
        f"{_pct(validation['top1_accuracy'])}(95% bootstrap CI "
        f"{_pct(validation['top1_ci95_low'])}-{_pct(validation['top1_ci95_high'])})였고, "
        f"다수 클래스 기준선 {_pct(validation['majority_baseline_accuracy'])} 대비 "
        f"{100 * float(validation['top1_lift_over_majority']):.2f}%p 높았다. "
        f"Macro Top-1은 {_pct(validation['macro_top1_accuracy'])}, ECE는 "
        f"{float(validation['ECE']):.3f}이었다.",
        "",
        f"판단보류 정책을 적용하면 검증 사건의 {_pct(validation['review_coverage'])}를 분석관 검토 대상으로 "
        f"제시했으며, 그 집합의 선택적 정확도는 {_pct(validation['selective_accuracy'])}, 전체 검증 사건 대비 "
        f"오추천률은 {_pct(validation['false_attribution_rate'])}였다. 이 수치는 공개 보고서 라벨의 품질과 "
        "클래스 불균형에 영향을 받으므로 수사기관 수준의 귀속 정확도로 해석하지 않는다.",
        "",
        "## 3. Ghostwriter 외부사례 적용",
        "",
        "개발 데이터와 분리해 리투아니아 NCSC의 2018·2019 사건 2건을 과거 참조 사건으로 구성하고, "
        "Mandiant 2021 기술 부록의 Polskie Radio, gift/VIDEOKILLER, SOCIS 사건 3건을 홀드아웃 질의로 사용하였다. "
        "발생일이 확인되지 않은 홀드아웃은 임의 날짜를 부여하지 않고 보고서 공개일을 시간 기준으로 사용하였다.",
        "",
        comparison.to_markdown(index=False),
        "",
        decisions.to_markdown(index=False),
        "",
        f"통합 순위에서 세 사건 모두 Ghostwriter/UNC1151 후보가 1위였고 Top-1은 "
        f"{_pct(external['top1_accuracy'])}였다. 다만 표본이 3건이고 실제 라벨이 한 행위자뿐이므로 "
        "다수 클래스 기준선도 100%이다. 따라서 이 결과는 일반화 성능이나 귀속 정확도의 입증이 아니라, "
        "공개 사건을 프레임워크 스키마와 점수·판단보류 절차에 적용할 수 있음을 보이는 사례 검증이다.",
        "",
        "Polskie Radio와 SOCIS는 과거 사건과 각각 88.99.132[.]118 및 94.103.82[.]136을 공유한다. "
        "gift/VIDEOKILLER 사건은 참조 사건과 직접 공유 IOC가 없어 IOC-only가 `no_signal`이었으나 "
        "서사·TTP·채널 근거로 통합 후보가 형성되었다. SOCIS는 정답 후보가 1위였음에도 보정 신뢰도가 "
        "기준보다 낮아 판단보류되었다.",
        "",
        "## 4. 해석 한계",
        "",
        "- 외부사례 3건은 단일 행위자 사례이므로 비교 벤치마크가 아니다.",
        "- 콘텐츠-only도 세 사건을 모두 1위로 검색했으므로 외부사례만으로 통합 방식의 우월성을 주장할 수 없다.",
        "- 공개 보고서의 서술과 IOC를 사후 정규화한 회고적 검증이며, 실시간 선행 예측 실험이 아니다.",
        "- 공유 IP는 강한 조사 단서지만 동일 행위자나 국가기관을 단독으로 입증하지 않는다.",
        "- `analyst_review`는 후속 조사 대상이라는 뜻이며 법적 귀속, 범죄 성립 또는 자연인 신원 확인이 아니다.",
        "",
        "## 5. 출처",
        "",
        "- [Lithuanian NCSC incident No. 152827](https://www.nksc.lt/doc/en/analysis/2018_01_29_Brief_review_of_an_incident_analysis.pdf)",
        "- [Lithuanian NCSC incident No. 163811](https://www.nksc.lt/doc/en/analysis/2019_04_30_Brief_targeted_attack_analysis.pdf)",
        "- [Mandiant Ghostwriter report page (2020)](https://cloud.google.com/blog/topics/threat-intelligence/ghostwriter-influence-campaign/)",
        "- [Mandiant Ghostwriter update (2021)](https://cloud.google.com/blog/topics/threat-intelligence/espionage-group-unc1151-likely-conducts-ghostwriter-influence-activity)",
        "",
        "세부 레코드와 해시는 `results/evidence_provenance.csv`에 기록하였다.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_generalization_validation_report(
    benchmark: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a transparent paper-ready summary of the frozen external test."""
    metric = benchmark["evaluation"].iloc[0]
    acceptance = benchmark["acceptance"]
    overall = bool(acceptance.loc[acceptance["criterion"] == "overall", "passed"].iloc[0])
    classes = benchmark["class_metrics"].copy()
    conditions = benchmark["condition_summary"].copy()
    predictions = benchmark["predictions"]
    errors = predictions[~predictions["top1_correct"]].copy()

    class_table = classes[[
        "actual_actor", "queries", "top1_accuracy", "top3_accuracy",
        "review_coverage", "selective_accuracy", "false_attribution_rate",
    ]].copy()
    condition_table = conditions[[
        "condition", "signal_coverage", "top1_accuracy_all_queries",
        "top1_accuracy_when_ranked",
    ]].copy()
    error_table = errors[[
        "query_event_id", "actual_actor", "predicted_actor", "actual_rank",
        "decision", "assessment_confidence",
    ]].copy()

    lines = [
        "# 다중 캠페인 외부 일반화 파일럿 결과",
        "",
        "## 1. 평가 설계",
        "",
        "평가 결과를 확인하기 전에 `generalization_protocol.yaml`에 데이터 구성, "
        "가중치, 판단보류 기준, 합격 기준을 고정하였다. Ghostwriter/UNC1151, "
        "Doppelganger, Spamouflage/Dragonbridge, Storm-1516/Neva Flood의 네 캠페인마다 "
        "과거 참조 사건 2건과 이후 홀드아웃 사건 5건을 배치하였다. 참조 자료와 "
        "홀드아웃 자료는 캠페인별로 출처 ID와 시간을 분리했으며 합성 IOC는 사용하지 않았다.",
        "",
        "홀드아웃의 행위자 라벨은 후보 프로필, 임베딩, IOC·TTP·채널 유사도, "
        "temperature 보정에 사용하지 않고 최종 채점에서만 사용하였다. 홀드아웃끼리 "
        "서로 참조 사건이 되는 경로도 코드에서 차단하였다. 실행 후 재튜닝은 하지 않았다.",
        "",
        f"프로토콜 SHA-256: `{benchmark['protocol_sha256']}`",
        "",
        "## 2. 전체 결과",
        "",
        f"20개 홀드아웃 사건은 모두 평가 가능했다. Top-1 정확도는 "
        f"{_pct(metric['top1_accuracy'])}(95% bootstrap CI "
        f"{_pct(metric['top1_ci95_low'])}-{_pct(metric['top1_ci95_high'])})였고, "
        f"다수 클래스 기준선 {_pct(metric['majority_baseline_accuracy'])}보다 "
        f"{100 * float(metric['top1_lift_over_majority']):.1f}%p 높았다. "
        f"Macro Top-1은 {_pct(metric['macro_top1_accuracy'])}, Top-3는 "
        f"{_pct(metric['top3_accuracy'])}, MRR은 {float(metric['MRR']):.3f}였다.",
        "",
        f"판단보류를 적용하면 전체 사건의 {_pct(metric['review_coverage'])}가 분석관 "
        f"검토 대상으로 남았고, 이 집합의 선택 정확도는 "
        f"{_pct(metric['selective_accuracy'])}, 전체 사건 대비 잘못된 검토 제안 비율은 "
        f"{_pct(metric['false_attribution_rate'])}였다. 사전 등록한 기준은 "
        f"{'모두 통과하였다' if overall else '일부 통과하지 못하였다'}.",
        "",
        "## 3. 조건별 비교",
        "",
        condition_table.to_markdown(index=False),
        "",
        "콘텐츠-only는 전체 사건에서 75%의 Top-1 정확도를 보였고, 실제 IOC가 "
        "참조 사건과 직접 연결된 경우가 적어 IOC-only의 신호 커버리지는 10%에 그쳤다. "
        "통합 방식은 Top-1 80%로 콘텐츠-only보다 5%p 높았다. 따라서 이 파일럿은 "
        "통합의 보완 효과를 보이지만, 큰 폭의 우월성을 입증한 결과로 해석해서는 안 된다.",
        "",
        "## 4. 클래스별 결과",
        "",
        class_table.to_markdown(index=False),
        "",
        "Ghostwriter, Doppelganger, Storm-1516은 각 5건 모두 1위에 올랐다. "
        "Spamouflage/Dragonbridge는 5건 중 1건만 1위였고 나머지 4건은 "
        "Doppelganger와 혼동되었다. 다만 네 건 모두 신뢰도·마진 기준을 통과하지 못해 "
        "시스템은 분석관 검토 제안 대신 판단보류했다. 상업형 위장 뉴스 사이트와 "
        "콘텐츠 신디케이션이 매체 사칭 캠페인과 의미·행위 특징상 겹친 것이 주요 원인이다.",
        "",
        "### 오분류 사건",
        "",
        error_table.to_markdown(index=False) if not error_table.empty else "오분류 없음.",
        "",
        "## 5. 해석 범위와 한계",
        "",
        "이 결과는 공개 보고서만으로 구성된 네 개의 알려진 후보 중 어느 캠페인의 "
        "과거 양상과 더 가까운지를 순위화할 수 있다는 폐쇄형 파일럿 근거다. 새로운 "
        "행위자 발견, 국가 배후 입증, 조직·개인의 신원 확인, 형사소송상 귀속 증명은 "
        "평가하지 않았다. 사건 설명은 공식 보고서를 수작업으로 정규화했으므로 작성자 "
        "편향 가능성이 있고, 클래스당 홀드아웃이 5건에 불과해 신뢰구간도 넓다.",
        "",
        "또한 관측일이 공개되지 않은 일부 사건은 보고서 발행일을 시간 기준으로 사용했다. "
        "따라서 자동 검사는 저장된 날짜상 선후관계를 확인한 것이며 모든 사건의 엄격한 "
        "발생시점 분리를 보장하지 않는다. Dragonbridge의 참조·홀드아웃은 보고서 ID는 "
        "다르지만 동일한 위협정보 조직 계열의 자료이고, Ghostwriter는 campaign-group, "
        "나머지는 campaign 수준이어서 분류 단위도 완전히 동일하지 않다.",
        "",
        "GLASSBRIDGE 보고서의 상업 유통업체와 위장 뉴스 사이트 전체를 Dragonbridge로 "
        "귀속한 것은 아니다. 해당 자료가 명시적으로 Dragonbridge 연계 콘텐츠라고 기술한 "
        "게시·유통 사례를 홀드아웃으로 정규화한 것이다. 특히 상업 유통 생태계 수준의 "
        "기술과 개별 콘텐츠 수준의 연결이 섞여 있어 이 클래스의 정답 라벨은 다른 세 "
        "클래스보다 더 신중하게 해석해야 한다.",
        "",
        "따라서 논문에서는 '일반화된 행위자 귀속 성능을 입증했다'보다 "
        "'출처·시간 분리된 다중 캠페인 환경에서 조사 후보 순위화의 외부 타당성을 "
        "예비 검증했다'고 표현하는 것이 타당하다. 후속 연구는 미등록 캠페인과 무관 "
        "사건을 포함한 open-set 거부 성능, 독립 코더 간 일치도, 더 큰 사건 표본을 다뤄야 한다.",
        "",
        "## 6. 공개 근거",
        "",
        "- [Meta CIB report (2022)](https://about.fb.com/news/2022/09/removing-coordinated-inauthentic-behavior-from-china-and-russia/)",
        "- [VIGINUM RRN technical report (2023)](https://www.sgdsn.gouv.fr/files/files/Publications/20230719_NP_VIGINUM_RAPPORT-CAMPAGNE-RRN_EN.pdf)",
        "- [Mandiant DRAGONBRIDGE report (2022)](https://cloud.google.com/blog/topics/threat-intelligence/prc-dragonbridge-influence-elections/)",
        "- [Google Threat Intelligence GLASSBRIDGE report (2024)](https://cloud.google.com/blog/topics/threat-intelligence/glassbridge-pro-prc-influence-operations)",
        "- [Microsoft MTAC report (2024)](https://blogs.microsoft.com/on-the-issues/2024/04/17/russia-us-election-interference-deepfakes-ai/)",
        "- [VIGINUM Storm-1516 technical report (2025)](https://www.sgdsn.gouv.fr/files/files/Publications/20250507_TLP-CLEAR_NP_SGDSN_VIGINUM_Technical%20report_Storm-1516.pdf)",
        "- [Mandiant Ghostwriter update (2021)](https://cloud.google.com/blog/topics/threat-intelligence/espionage-group-unc1151-likely-conducts-ghostwriter-influence-activity)",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
