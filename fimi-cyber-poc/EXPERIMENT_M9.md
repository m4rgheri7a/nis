# M9 — 증거 구조화 조건 비교 실험 결과

`results/`는 gitignore되므로 이 문서가 수치의 저장소 내 정본이다. 논문 본문·표는 아직
수정하지 않았다. 아래 수치와 "주장 가능 범위"를 근거로 갱신하면 된다.

## 1. 이 실험이 답하는 질문

기존 PoC에서 Qwen3 증거 구조화 실험과 FCLS 후보 순위화 파이프라인은 연결되어 있지 않았다.
발표된 Top-1 0.80 / Top-3 1.00 / MRR 0.8833은 **사람이 정규화한 필드**로 계산된 값이며,
LLM의 기여로 볼 근거가 없었다.

M9는 사례 원문에서 증거를 다시 도출해 **동일한 다운스트림**(증거 그래프 → FCLS →
후보 순위화 → 판정 보류)으로 순위화한다. 따라서 순위 지표의 차이를 추출 방식에 귀속할 수 있다.

| 조건 | 증거 출처 |
|------|-----------|
| `curated_oracle` | 사람이 정규화한 필드 — 성능 상한이자 기존 발표 수치의 조건 |
| `rules_only` | case dossier에서 키워드·정규식 추출 |
| `llm_guarded` | 동일 dossier에서 Qwen3-14B 구조화 + 가드레일 |
| `llm_only` | 동일 LLM 출력, 가드레일 해제 — 대조군 |

## 2. 실행 조건

| 항목 | 값 |
|------|-----|
| 모델 | `qwen3:14b` (Ollama, digest `bdbd181c33f2`, 9.3GB, Q4_K_M) |
| 디코딩 | temperature 0, seed 42, non-thinking, num_predict 512 |
| GPU | NVIDIA RTX 5070 12GB — 모델 전량 GPU 상주 (100% GPU, 9.6GB) |
| CPU / RAM | Ryzen 5 9600X / 25GB (WSL2) |
| 임베딩 | SBERT `paraphrase-multilingual-mpnet-base-v2` |
| 보정 온도 | 0.04026 (메인 파이프라인이 시간순 보정으로 적합) |
| 데이터 | 28개 사건 (reference 8, holdout 20), 4개 캠페인 |
| 추출 소요 | 28개 사건당 약 108초 |
| 테스트 | 73 passed |

> todo.md에 기록된 환경(Qwen3-8B / RTX 4060 Ti 8GB / RAM 32GB)은 다른 PC 기준이다.
> 현재 장비는 VRAM이 더 커서 14B를 전량 GPU에 올릴 수 있으므로 14B로 실행했다.
> 모델 태그는 `--model` 파라미터이므로 8B·30B-A3B 재실행은 명령어만 바꾸면 된다.

## 3. 후보 순위화 결과 (holdout 20건)

| 조건 | Top-1 | Top-3 | MRR | 검토 커버리지 | 판정 보류 | 선택 정확도 | **오귀속률** |
|------|-------|-------|-----|--------------|----------|------------|------------|
| curated_oracle | 0.80 | 1.00 | 0.883 | 0.75 | 0.25 | 1.000 | **0.00** |
| rules_only | 0.75 | 0.90 | 0.842 | 0.80 | 0.20 | 0.875 | **0.10** |
| **llm_guarded** | **0.80** | **1.00** | **0.892** | 0.75 | 0.25 | 0.867 | **0.10** |
| llm_only | 0.85 | 1.00 | 0.925 | 0.40 | 0.60 | 1.000 | **0.00** |

## 4. 증거 추출 품질 (큐레이션 정답 대비)

| 조건 | TTP F1 | 채널 F1 | 표적 F1 | 국가 F1 | IOC recall | IOC precision | 환각 IOC |
|------|--------|---------|---------|---------|-----------|--------------|---------|
| rules_only | 0.492 | 0.710 | 0.771 | 0.607 | 0.90 | 0.875 | 0 |
| llm_guarded | 0.523 | 0.710 | 0.771 | 0.607 | 1.00 | 0.975 | 0 |
| llm_only | 0.464 | 0.471 | 0.551 | 0.730 | 1.00 | 1.000 | 0 |

채널·표적·국가 F1이 rules_only와 llm_guarded에서 동일한 것은 설계상 의도된 결과다. 이 세
계열은 결정론적 규칙 결과만 증거 그래프로 승격한다(todo §10). LLM은 TTP와 IOC 후보에만
기여한다.

## 5. 캠페인별 Top-1

| 캠페인 | oracle | rules_only | llm_guarded | llm_only |
|--------|--------|-----------|-------------|----------|
| Doppelganger | 1.0 | 1.0 | 1.0 | 1.0 |
| Ghostwriter/UNC1151 | 1.0 | 1.0 | 1.0 | 1.0 |
| **Spamouflage/Dragonbridge** | **0.2** | **0.0** | **0.2** | 0.6 |
| Storm-1516/Neva Flood | 1.0 | 1.0 | 1.0 | 0.8 |

오귀속은 전부 Spamouflage/Dragonbridge에서 발생한다(rules_only·llm_guarded 각 0.4).

## 6. IOC 희소성 ablation — 기술부록 제거 (`--no-annex`)

사건 요약문만 입력했을 때 (`results/*_no_annex.*`):

| 조건 | Top-1 | Top-3 | MRR | IOC recall | **환각 IOC** |
|------|-------|-------|-----|-----------|-------------|
| curated_oracle | 0.80 | 1.00 | 0.883 | — | — |
| rules_only | 0.75 | 0.95 | 0.846 | **0.00** | 0 |
| llm_guarded | 0.75 | 1.00 | 0.867 | **0.00** | **2 (전부 차단됨)** |
| llm_only | 0.70 | 1.00 | 0.850 | **0.00** | **2 (전부 그래프 진입)** |

부록이 없으면 IOC recall이 0으로 붕괴한다. 기존 실험의 "IOC candidate coverage 0.036"은
모델 성능 문제가 아니라 **입력 텍스트에 지표가 없기 때문**임이 확인된다.

그리고 이때 **환각이 처음 발생한다.** `gw-test-2021-polskie-radio`에서 Qwen3-14B는
지표가 없는 텍스트를 받자 `1.2.3.4`와 `example.com`을 지어냈다. 가드레일은 둘 다 거부했고
(`ioc_candidates=[]`), llm_only는 둘 다 증거 그래프에 넣었다.

## 7. 주장 가능 범위

**말할 수 있는 것**

- Qwen3-14B 구조화 결과를 가드레일과 함께 메인 FCLS 파이프라인에 연결하면, 사례 원문만
  읽고도 큐레이션 상한과 동일한 Top-1(0.80)·Top-3(1.00)에 도달했고 MRR은 0.883 → 0.892로
  소폭 높았다.
- 동일 조건에서 규칙 기반 추출만 쓰면 Top-1 0.75 / Top-3 0.90 / MRR 0.842에 그친다.
  따라서 **LLM 구조화는 규칙 기반 대비 최종 후보 순위화에 실질적으로 기여했다.**
- 기여의 출처는 두 가지로 특정된다: TTP F1 0.492 → 0.523, 그리고 정규식이 볼 수 없는
  계정 핸들(`intrusion_trutl`)을 회수해 IOC recall 0.90 → 1.00.
- 가드레일은 지표가 없는 입력에서 실제로 발생한 환각 IOC 2건을 전부 차단했다.

**말하면 안 되는 것**

- "Qwen3가 FCLS 성능을 향상했다"를 일반화하지 말 것. 20개 holdout, 4개 캠페인, 폐쇄형
  후보 집합에서의 결과다. Top-1 차이 0.05는 사건 1건에 해당한다.
- **`llm_guarded`의 오귀속률은 0.10으로, 큐레이션 상한(0.00)보다 나쁘다.** 추출로 도출한
  증거는 자동 판정을 허용해선 안 되는 사건 2건을 통과시켰다. 판정 보류 정책이
  Spamouflage/Dragonbridge에서 작동하지 않는다. 이 사실을 숨기지 말 것.
- `llm_only`가 Top-1 0.85로 가장 높지만 이를 "가드레일이 불필요하다"로 읽지 말 것.
  이 조건은 60%를 판정 보류로 넘겨(검토 커버리지 0.40) 쉬운 사건만 자동 처리한 결과이고,
  통제 어휘를 벗어난 자유형 라벨을 내놓아 채널 F1 0.710 → 0.471, 표적 F1 0.771 → 0.551로
  분석관이 쓸 수 있는 온톨로지가 무너진다. 부록이 없으면 환각을 그대로 그래프에 넣는다.
- 부록이 있는 조건에서는 **어떤 조건에서도 환각 IOC가 0건**이었다. "가드레일이 환각을
  막았다"는 주장은 `--no-annex` 조건에 한정해서만 성립한다.
- 한국어 성능은 여전히 미검증이다. 데이터는 전부 영어다.

## 8. 재현 절차

```bash
cd fimi-cyber-poc
python -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama pull qwen3:14b

.venv/bin/python scripts/run_all.py --llm-model qwen3:14b        # M1~M9 전체
.venv/bin/python scripts/run_condition_benchmark.py --model qwen3:14b
.venv/bin/python scripts/run_condition_benchmark.py --model qwen3:14b \
    --no-annex --output-suffix no_annex                           # IOC 희소성 ablation
.venv/bin/python -m pytest -q                                     # 73 passed
```

`run_condition_benchmark.py`는 `results/attribution_calibration.csv`에서 보정 온도를 읽으므로
`run_all.py`를 먼저 한 번 실행해야 한다.

## 9. 누출 통제

네 조건이 읽는 case dossier는 보고서 요약문과 공개 기술부록(defang 표기)을 복원한
텍스트다(`src/fimicyber/llm/dossier.py`). 다음이 테스트로 강제된다
(`tests/test_m9_condition_benchmark.py`):

- 28개 dossier 어디에도 행위자·캠페인 별칭이 없다.
- 큐레이션 정답 필드(TTP·채널·표적·IOC·campaign_id·reported_actor)는 렌더링되지 않는다.
- 보고서 URL은 슬러그에 캠페인명을 담으므로 dossier에 넣지 않는다.
- 네트워크 IOC는 defang 표기로만 출력된다.
- 추출이 빈 결과를 내면 사건 필드도 비어야 한다 — 정답 필드로 되돌아가지 않는다.

기술부록은 공개 보고서의 IOC 표를 텍스트로 **재구성**한 것이지 원본 PDF의 축자 복사가
아니다. 이 점은 논문에 명시해야 한다.

## 10. 이번에 고친 결함

1. **`apply_structured_evidence()`의 gold fallback (데이터 누출).**
   `item.ttps or event.ttps` 패턴이 추출 실패 시 큐레이션 정답을 조용히 복원했다.
   이대로 조건 비교를 돌리면 추출 조건이 오라클 필드를 물려받아 점수가 부풀려진다.
   `replace` 시맨틱으로 교체했다.
2. **IOC 가드레일의 defang 미대응.** 원문은 `88[.]99[.]132[.]118`, 모델 응답은
   `88.99.132.118`이라 진짜 IOC가 환각으로 오인되어 버려졌다. refang한 텍스트에도
   대조하도록 고쳤다.
3. **가드레일이 LLM의 IOC 기여를 무효화.** 병합식이 `rules ∪ (llm ∩ rules)`여서 항상
   `rules`와 같았다. 가시성 검증으로 바꿔 정규식이 못 보는 계정 핸들을 회수한다.
4. **dossier 부록 헤더가 IOC 분류기를 오염.** "published report"라는 표현이
   `classify_ioc`의 출처-키워드 휴리스틱을 발동시켜 진짜 C2 IP·도메인을
   `EvidenceSourceURL`로 오분류했다. 헤더 문구를 바꿔 gold IOC 회수율이 0.583 → 0.90이 됐다.
5. **`target_countries` 미추출.** 표적 계열(가중치 0.10)의 절반이 모든 조건에서
   큐레이션 필드로 남아 있었다. 국가·지명 gazetteer를 추가해 추출 조건에서 재도출한다.
