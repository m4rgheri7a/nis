# FIMI-Cyber Link Score and Attribution-Support PoC

기존 관련 사건 검색(FCLS)에 공개자료 기반 행위자 귀속지원 계층을 추가했다. 귀속지원은
query보다 먼저 관측된 사건만 reference profile로 사용하며, `reported_actor`는 입력 특징이
아니라 시간 분리 평가 라벨로만 사용한다. 합성 IOC는 귀속 점수에서 제외된다.

추가 산출물:

- `results/attribution_hypotheses.csv`: 전체 후보 순위, 경쟁 가설, 성분 분해, 신뢰도 상한
- `results/attribution_evaluation.csv`: Top-1/Top-3, macro 정확도, 다수 클래스 기준선, MRR, Brier, ECE
- `results/attribution_graph.json`: query → reference event → actor hypothesis 증거 경로
- `results/attribution_calibration.csv`: 시간순 temperature 보정 구간과 NLL
- `results/attribution_error_analysis.csv`: 정답·오답 및 판단보류 사례 분석
- `results/external_ghostwriter_*.csv`: 개발 데이터와 분리한 Ghostwriter 외부사례 결과
- `results/evidence_provenance.csv`: 사건·IOC별 출처 ID와 레코드 SHA-256
- `results/paper_ready_attribution_validation.md`: 과장 방지 문구를 포함한 논문 삽입용 결과 요약
- `results/figures/external_ghostwriter_evidence.{png,svg}`: 외부사례 증거 경로와 판단보류 시각화
- `results/generalization_*.csv`: 4개 캠페인, 20개 홀드아웃의 봉인 외부 평가 결과
- `results/paper_ready_generalization_validation.md`: 다중 캠페인 결과와 한계의 논문 삽입용 요약
- `results/figures/generalization_benchmark.{png,svg}`: 조건·클래스·혼동행렬 시각화

다중 캠페인 평가는 `data/external/generalization_protocol.yaml`에 데이터 구성, 가중치,
판단보류 기준과 합격 기준을 먼저 고정한다. 홀드아웃 라벨은 최종 채점에만 사용하고,
명시된 reference 사건 외의 사건은 후보 프로필에 포함하지 않는다.

이 출력은 분석관의 후속 조사를 지원하는 가설이며 범죄, 신원 또는 법적 귀속을 확정하지 않는다.

허위정보 사건 간 연계 가능성을 내러티브 유사도(SBERT)와 사이버 IOC 연결성(증거 그래프)의 통합 점수(FCLS)로 산정하는 PoC.

## 실행

```bash
pip install -r requirements.txt
python scripts/run_all.py          # 전체 파이프라인 실행
```

```bash
python scripts/serve.py            # 웹 대시보드 → http://localhost:5000
```

```bash
python scripts/run_all.py --dry-run  # 단계 목록만 출력
pytest tests/ -v                     # 자동화 테스트 69개
```

### LLM 증거 구조화

LLM은 행위자를 확정하지 않고 비정형 사례 설명에서 근거 문장과 TTP 후보를 구조화한다.
채널·표적·IOC는 결정적 검증층이 원문 가시성과 통제 어휘를 확인한 뒤에만 증거 객체로
승격한다. 백엔드는 로컬 Ollama이며 모델 태그는 파라미터다.

```bash
ollama pull qwen3:14b
python scripts/run_llm_hfes.py --model qwen3:14b --output-suffix qwen3_14b
```

이 스크립트는 구조화 품질만 채점한다. 구조화 결과가 실제 후보 순위화에 기여하는지는
아래 조건 비교 실험이 측정한다.

### 증거 구조화 조건 비교 (M9)

구조화 모듈이 최종 후보 순위화에 실제로 기여하는지 확인하려면, 큐레이션된 정답 필드가
아니라 사례 원문에서 증거를 다시 도출해 동일한 다운스트림으로 순위화해야 한다.
`run_condition_benchmark.py`는 하나의 데이터 분할과 하나의 다운스트림(증거 그래프 →
FCLS → 후보 순위화 → 판정 보류)에 네 조건을 통과시킨다.

| 조건 | 증거 출처 |
|------|-----------|
| `curated_oracle` | 사람이 정규화한 필드 (성능 상한, 기존 발표 수치의 조건) |
| `rules_only` | 사례 dossier에서 규칙·정규식 추출 |
| `llm_guarded` | 동일 dossier에서 LLM 구조화 + 가드레일 |
| `llm_only` | 동일 LLM 출력, 가드레일 해제 (실패 사례 대조군) |

```bash
python scripts/run_all.py                          # M9는 LLM 없이 2조건 실행
python scripts/run_all.py --llm-model qwen3:14b    # 4조건 전부
python scripts/run_condition_benchmark.py --model qwen3:14b
python scripts/run_condition_benchmark.py --no-annex   # 기술부록 제거 → IOC 희소성 측정
```

**누출 통제.** 네 조건 모두 동일한 case dossier를 읽는다. dossier는 보고서 요약문과
공개 기술부록(defang 표기 IOC)을 복원한 텍스트이며, 행위자·캠페인 명칭은 스크러빙된다
(`llm/dossier.py`). 큐레이션 정답 필드(TTP·채널·표적·IOC·campaign_id·reported_actor)는
dossier에 렌더링되지 않고, 홀드아웃 라벨은 순위화가 끝난 뒤 채점에만 열린다.
`tests/test_m9_condition_benchmark.py`가 이 계약을 강제한다.

기술부록은 공개 보고서의 IOC 표를 텍스트로 재구성한 것이지 원본 PDF의 축자 복사가 아니다.

산출물: `results/condition_ranking_metrics.csv`, `condition_extraction_metrics.csv`,
`condition_hallucinated_iocs.csv`, `condition_case_dossiers.jsonl`,
`condition_run_manifest.json`(모델 태그·digest·seed·GPU), `condition_benchmark_summary.md`.

## 구조

```
config/          weights.yaml, synthetic_scenarios.yaml
src/fimicyber/
  schema.py      IOC / Event pydantic 모델
  loaders/       DISINFOX 파서 (fixture 폴백 포함)
  ioc/           추출 → 분류 → confidence → 합성 생성
  nlp/           SBERT 임베딩 + N(i,j)
  graph/         증거 그래프 + I_direct / I_path
  scoring/       D·C·T·A 성분, FCLS 재정규화, Priority
  eval/          GT, MAP/nDCG, E1/E2/E3, ablation, 강건성
  viz/           pyvis 증거 경로, matplotlib 차트, report.md
  llm/           case dossier 생성·라벨 스크러빙, 가드레일 증거 구조화
scripts/
  run_all.py        전체 파이프라인 단일 진입점 (M9 조건 비교 포함)
  run_llm_hfes.py   LLM 증거 구조화 품질 채점
  run_condition_benchmark.py  4조건 비교 — 구조화 → 그래프 → FCLS → 후보 순위화
  serve.py          Flask 웹 대시보드 (파이프라인 실행·테스트·결과 시각화)
  make_fixtures.py  샘플 20건 생성기 (DISINFOX 미사용 시 자동 호출)
```

## 결과물

| 파일 | 내용 |
|------|------|
| `results/pairwise_scores.csv` | N·I·D·C·T·A·FCLS_E1/E2/E3 전 사건쌍 |
| `results/metrics_summary.csv` | P@k, MAP, nDCG@10, 95% CI |
| `results/ablation.csv` | 성분별 제거 실험 |
| `results/gridsearch.csv` | α×β 그리드 탐색 |
| `results/robustness.csv` | noise×coverage 9시나리오 |
| `results/priority_table.csv` | 사건별 조사 우선순위 |
| `results/figures/` | 막대·선·히트맵·레이더·분포 차트 (7종) |
| `results/evidence_paths/` | 상위 3쌍 pyvis HTML |
| `results/report.md` | 자동 생성 종합 리포트 |

## 설계 원칙

- 모든 수치는 `config/weights.yaml`에서만 읽음 (하드코딩 금지)
- 합성 IOC는 예약 대역만 사용 (`.test` / RFC5737 IP / RFC5398 ASN)
- 평가 시 A항(actor) ζ=0 강제 — 순환논리 방지
- SBERT 설치 실패 시 TF-IDF 자동 폴백, 폴백 내역은 `report.md`에 기록
