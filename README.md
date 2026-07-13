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
pytest tests/ -v                     # 자동화 테스트 52개
```

### Qwen3-8B 증거 구조화

LLM은 행위자를 확정하지 않고 비정형 사례 설명에서 근거 문장과 TTP 후보를 구조화한다.
채널·표적·IOC는 결정적 검증층이 원문 가시성과 통제 어휘를 확인한 뒤에만 증거 객체로
승격한다. 기본 백엔드는 로컬 Ollama의 `qwen3:8b`이다.

```bash
ollama pull qwen3:8b
python scripts/run_llm_hfes.py --output-suffix qwen3_8b
```

결과는 `results/llm_structured_evidence_qwen3_8b.jsonl`,
`results/llm_structuring_evaluation_qwen3_8b.csv`,
`results/llm_structuring_summary_qwen3_8b.md`에 저장된다.

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
scripts/
  run_all.py        전체 파이프라인 단일 진입점
  run_llm_hfes.py   Qwen3-8B 증거 구조화 및 결정적 검증
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
