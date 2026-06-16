# FIMI-Cyber Link Score PoC

허위정보 사건 간 연계 가능성을 내러티브 유사도(SBERT)와 사이버 IOC 연결성(증거 그래프)의 통합 점수(FCLS)로 산정하는 PoC.

## 실행

```bash
pip install -r requirements.txt
python scripts/run_all.py
```

```bash
python scripts/run_all.py --dry-run  # 단계 목록만 출력
pytest tests/ -v                     # 골든 테스트 27개
```

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
  run_all.py     전체 파이프라인 단일 진입점
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
| `results/figures/` | 막대·선·히트맵 차트 |
| `results/evidence_paths/` | 상위 3쌍 pyvis HTML |
| `results/report.md` | 자동 생성 종합 리포트 |

## 설계 원칙

- 모든 수치는 `config/weights.yaml`에서만 읽음 (하드코딩 금지)
- 합성 IOC는 예약 대역만 사용 (`.test` / RFC5737 IP / RFC5398 ASN)
- 평가 시 A항(actor) ζ=0 강제 — 순환논리 방지
- SBERT 설치 실패 시 TF-IDF 자동 폴백, 폴백 내역은 `report.md`에 기록
