# FIMI-Cyber Link Score PoC — 구현 명세서 (v1.0)

논문 「외국발 허위정보와 사이버 IOC 연계 분석 프레임워크」 Ⅳ장(PoC)의 구현 명세.
이 문서를 받은 Claude Code는 아래 지침과 명세에 따라 저장소를 처음부터 구축한다.

---

## 0. Claude Code 작업 지침 (필수 준수)

1. **마일스톤 순서 준수**: 12장의 M0→M8 순서로 구현한다. 각 마일스톤의 DoD(Definition of Done)를 통과한 뒤에만 다음으로 진행한다. 커밋 단위 = 마일스톤.
2. **하이퍼파라미터 하드코딩 금지**: 모든 수치(가중치, τ, λ 등)는 `config/weights.yaml`에서만 읽는다. 기본값은 16장 YAML 전문을 그대로 사용한다.
3. **결정론 보장**: 모든 난수는 config의 `seed`(기본 42)로 고정. 같은 config → 같은 결과.
4. **단일 명령 재현**: `python scripts/run_all.py` 한 번으로 데이터 적재→점수→실험→그림→리포트까지 전부 생성되어야 한다.
5. **골든 테스트 필수**: 13장에 명시된 수치를 pytest로 박제한다. 수식 구현이 골든 값과 다르면 구현이 틀린 것이다.
6. **네트워크 실패 시 폴백**: 15장의 폴백 경로를 따르고, 어떤 폴백을 썼는지 `README.md`와 `results/report.md`에 반드시 기록한다. 임의로 다른 대체물을 쓰지 않는다.
7. **합성 IOC는 예약 대역만 사용**: 4.6절 규칙. 실존 가능한 도메인/IP/ASN 생성은 절대 금지 (오귀속 방지 — 논문의 핵심 윤리 원칙).
8. **평가 시 A항 제외 하드 가드**: 순환논리 방지를 위해 성능 평가에서는 ζ=0을 코드로 강제한다 (10.3절).
9. **PoC 품질 기준**: 과도한 추상화 금지. 클래스보다 순수 함수 우선, 타입 힌트 필수, 모듈당 책임 1개.
10. **산출물 파일명 고정**: 11장의 경로·파일명을 정확히 따른다 (논문 표·그림과 1:1 대응).

---

## 1. 프로젝트 개요

### 1.1 목적
허위정보 사건 간 연계 가능성을 **내러티브 유사도(SBERT)** 와 **사이버 IOC 연결성(증거 그래프)** 의 통합 점수(FCLS)로 산정하고, 통합 방식이 단일 축 방식보다 "관련 사건 검색" 성능이 우수함을 실험으로 검증한다.

### 1.2 검증 가설
- **H1**: 통합 분석(E3)이 콘텐츠-only(E1), IOC-only(E2)보다 관련 사건 검색 성능(MAP, nDCG)이 높다.
- **H2**: IOC 노이즈(공유 호스팅 등 저식별력 지표) 비율이 증가할 때 E2 성능은 급락하지만 E3는 완만하게 저하된다 (강건성).

### 1.3 최종 산출물
성능 비교표(E1/E2/E3), 강건성 곡선, 가중치 민감도 히트맵, ablation 표, 상위 사건쌍 증거 경로 시각화(HTML), 자동 생성 리포트(`results/report.md`). 전부 논문 Ⅳ장에 직접 인용 가능한 형태.

---

## 2. 기술 스택

- Python 3.11+ / GPU 불필요 (CPU로 충분)
- `requirements.txt`:

```
pandas>=2.0
numpy>=1.26
pydantic>=2.5
PyYAML>=6.0
sentence-transformers>=2.7
torch>=2.1
scikit-learn>=1.4
networkx>=3.2
pyvis>=0.3.2
matplotlib>=3.8
tldextract>=5.0
tqdm>=4.66
pytest>=8.0
```

- torch는 CPU wheel로 설치해도 무방. 설치가 무겁거나 실패하면 15장 TF-IDF 폴백 사용.

---

## 3. 저장소 구조

```
fimi-cyber-poc/
├── config/
│   ├── weights.yaml              # 16장 전문 그대로
│   └── synthetic_scenarios.yaml  # 17장 전문 그대로
├── data/
│   ├── raw/                      # DISINFOX 클론/원본 (git-ignore)
│   ├── curated/
│   │   ├── real_iocs.csv         # 사람이 채우는 입력 (비어 있어도 동작해야 함)
│   │   ├── ioc_relations.csv
│   │   ├── campaign_map.csv
│   │   └── benign_extra.txt
│   ├── interim/                  # events.jsonl, iocs.jsonl, mapping_report.md
│   └── processed/                # embeddings.parquet, graph.json
├── src/fimicyber/
│   ├── schema.py                 # pydantic 모델 (4.2)
│   ├── loaders/disinfox.py       # 4.1
│   ├── ioc/
│   │   ├── extract.py            # 정규식 + defang (4.4)
│   │   ├── classify.py           # 4분류 + 오탐 제거 (4.4)
│   │   ├── confidence.py         # IOCConfidence (4.5)
│   │   └── synthetic.py          # 합성 생성기 (4.6)
│   ├── nlp/
│   │   ├── embed.py              # 청킹 + 임베딩 + 캐시 (5장)
│   │   └── narrative.py          # N(i,j) (5장)
│   ├── graph/
│   │   ├── build.py              # 증거 그래프 (6장)
│   │   └── ioc_score.py          # I_direct, I_path (6장)
│   ├── scoring/
│   │   ├── components.py         # D, C, T, A (7장)
│   │   ├── fcls.py               # 결합 + 재정규화 (8장)
│   │   └── priority.py           # Priority(i) (9장)
│   ├── eval/
│   │   ├── groundtruth.py        # 10.1
│   │   ├── metrics.py            # P@k, MAP, nDCG (10.2)
│   │   └── experiments.py        # E1~E3, ablation, grid, robustness (10.3~10.5)
│   └── viz/
│       ├── evidence_path.py      # pyvis (11장)
│       └── charts.py             # matplotlib (11장)
├── scripts/
│   ├── run_all.py                # 전체 파이프라인 (--dry-run 지원)
│   └── make_fixtures.py          # 폴백용 샘플 사건 생성 (15장)
├── tests/                        # 13장
│   └── fixtures/
├── results/                      # 11장 산출물 (git-ignore, figures/ 포함)
├── requirements.txt
└── README.md
```

---

## 4. 데이터 계층

### 4.1 DISINFOX 로더 (`loaders/disinfox.py`)

**절차**
1. `git clone https://github.com/CyberDataLab/disinfox data/raw/disinfox`
2. 저장소를 탐색해 초기 적재용 사건 데이터 파일을 찾는다. README상 setup 스크립트가 "dataset of disinformation incidents"를 로드하므로, 해당 스크립트가 참조하는 JSON/CSV/BSON 파일을 역추적한다. 약 100~120건의 사건이 존재해야 정상.
3. 파일을 직접 파싱한다 (Docker/API 기동은 차선책 — 파일 파싱이 실패할 때만).

**어댑터 계약**
```python
def load_events(raw_dir: Path, cfg: Config) -> list[Event]:
    """DISINFOX 원본 → 내부 Event 스키마(4.2) 리스트.
    필드가 예상과 다르면 최선 매핑 후 결정 사항을
    data/interim/mapping_report.md에 기록한다."""
```

**매핑 규칙** (원본 필드명은 실제 확인 후 조정, 결정은 mapping_report.md에 기록)

| DISINFOX 개념 | 내부 필드 | 비고 |
|---|---|---|
| incident 제목/설명 | `title`, `description` | description이 Content이자 임베딩 대상 |
| 같은 campaign 관계 | `campaign_id` | **Ground Truth의 원천** — 반드시 보존 |
| threat actor | `reported_actor` | 맥락 속성. 점수 A항에만 사용, 평가 시 제외 |
| DISARM TTP 목록 | `ttps: list[str]` | 기법 ID 문자열 (예: "T0022") |
| 표적 국가 | `target_countries: list[str]` | ISO 코드로 정규화 |
| 날짜 | `first_seen`, `last_seen` | 단일 날짜면 둘 다 동일값 |
| 출처 URL | `evidence_sources` | EvidenceSource로 저장 (작전 IOC 아님) |

**주의**: campaign 관계가 별도 컬렉션/필드로 존재할 수 있음. 사건 상세에 "같은 캠페인의 다른 사건" 정보가 있으므로 이를 `campaign_id`로 역정규화한다. campaign 정보가 전혀 없으면 `reported_actor` 동일 여부를 campaign_id 대용으로 쓰되, 이 경우 평가의 ζ=0 가드가 더욱 중요함을 mapping_report.md에 명시.

### 4.2 공통 스키마 (`schema.py`, pydantic)

```python
class IOC(BaseModel):
    value: str                      # 정규화(refang)된 값
    ioc_type: Literal["domain","url","ipv4","email","hash_md5",
                      "hash_sha1","hash_sha256","ns","asn",
                      "account","tg_channel"]
    category: Literal["EvidenceSourceURL","PlatformContentURL",
                      "OperationalIOC","BenignReference"]
    confidence: float               # IOCConfidence, 0~1
    conf_components: dict[str,float]  # context/source/corroboration/type/freshness
    first_seen: date | None
    last_seen: date | None
    sources: list[str]              # 출처 URL 목록
    status: Literal["candidate","validated","rejected","needs_review"]
    synthetic: bool = False

class Event(BaseModel):
    event_id: str
    title: str
    description: str
    campaign_id: str | None         # GT 원천
    reported_actor: str | None      # 맥락 전용
    target_countries: list[str] = []
    target_sectors: list[str] = []
    first_seen: date | None
    last_seen: date | None
    ttps: list[str] = []
    channels: list[str] = []        # 플랫폼/채널명 (소문자 정규화)
    evidence_sources: list[str] = []  # 출처 URL
    iocs: list[IOC] = []            # OperationalIOC만 점수에 사용
    source_dataset: str = "disinfox"
```

저장: `data/interim/events.jsonl` (1행 1사건), `data/interim/iocs.jsonl`.

### 4.3 큐레이션 입력 파일 (human-in-the-loop)

사람이 공개 보고서에서 손으로 채우는 파일. **비어 있어도 전체 파이프라인은 합성 IOC만으로 완주해야 한다.**

**`data/curated/real_iocs.csv`**

| 컬럼 | 설명 |
|---|---|
| ioc_value | defang 표기 허용 (로더가 refang) |
| ioc_type | 4.2의 enum 값 |
| campaign | 자유 텍스트 키 (campaign_map.csv로 사건 매칭) |
| first_seen / last_seen | YYYY-MM-DD, 미상이면 공란 |
| source_url / source_org | 출처 보고서 |
| context_label | operational / platform_content |
| notes | 자유 기술 |

예시 행 (Qurium의 Doppelganger 기술 보고서에 공개된 실제 도메인):
```csv
ioc_value,ioc_type,campaign,first_seen,last_seen,source_url,source_org,context_label,notes
bild[.]eu[.]com,domain,doppelganger,2022-06-01,2022-09-30,https://www.qurium.org/alerts/under-the-hood-of-a-doppelganger/,Qurium,operational,Bild 클론 도메인
```

**`data/curated/ioc_relations.csv`** — 보고서에 명시된 인프라 관계만 입력 (라이브 DNS/WHOIS 조회 금지, 14장)

| 컬럼 | 값 |
|---|---|
| src_value / dst_value | IOC 또는 인프라 객체 값 |
| relation | resolves_to / uses_ns / belongs_to_asn / redirects_to / registered_at |
| source_url | 근거 보고서 |
| confidence | 0~1 (기본 0.9) |

**`data/curated/campaign_map.csv`** — real_iocs의 campaign 키 ↔ 사건 매칭 규칙

| 컬럼 | 값 |
|---|---|
| campaign_key | 예: doppelganger |
| match_mode | campaign_id / title_regex |
| pattern | campaign_id 값 또는 제목 정규식 (예: `(?i)doppelg`) |

### 4.4 IOC 추출·분류 (`ioc/extract.py`, `ioc/classify.py`)

**추출 대상**: `Event.description` 텍스트, curated CSV. (출처 웹문서 크롤링은 비범위 — URL 자체만 EvidenceSource로 저장)

**defang 복원(refang) 규칙**: `hxxp→http`, `[.]→.`, `(.)→.`, `{.}→.`, `[dot]→.`, `[at]→@`, `(@)→@`. 최종 보고서/시각화 출력 시에는 역방향(defang)으로 재변환하는 유틸도 제공.

**정규식** (refang 후 적용)
- ipv4: `\b(?:\d{1,3}\.){3}\d{1,3}\b` + 각 옥텟 0~255 검증
- url: `https?://[^\s"'<>)\]]+`
- domain: URL이 아닌 토큰 중 `tldextract`로 유효 TLD 확인된 것
- email: 표준 패턴
- hash: hex 32자(md5)/40자(sha1)/64자(sha256), 단어 경계
- tg_channel: `t.me/` 경로 (`@handle` 단독은 오탐이 많아 제외)

**4분류 규칙** (우선순위 순)
1. benign 목록 매치 → `BenignReference`
2. 주요 소셜 플랫폼 도메인(x/twitter, facebook, youtube, tiktok, t.me, vk, instagram, reddit)의 콘텐츠 URL → `PlatformContentURL`
3. Event.evidence_sources에 이미 있는 URL과 동일 → `EvidenceSourceURL`
4. 주변 문맥(±120자)에 작전 키워드 존재 → `OperationalIOC` (status=candidate)
   - 작전 키워드: indicator, ioc, phishing, malware, c2, command-and-control, spoofed, typosquat, clone(d), registered, hosted, redirect(s/ed) to, infrastructure, fake site/domain/outlet, impersonat*
   - 출처 키워드(source, report, according to, reference, archive, published by)가 우세하면 → `EvidenceSourceURL`
5. 문맥 불명 → `OperationalIOC` + `status=needs_review` + 낮은 C_context

**내장 benign 목록** (`ioc/classify.py`에 상수 + `benign_extra.txt`로 확장): 주요 플랫폼·검색엔진·주요 언론(bbc, reuters, nyt 등)·보안업체(mandiant, qurium, disinfo.eu 등)·정부 도메인(*.gov, *.europa.eu)·공용 DNS(8.8.8.8 등)·RFC1918 사설 대역·URL 단축 도메인(bit.ly 등 — 제거하지 않되 type_weight 최저로).

### 4.5 IOCConfidence (`ioc/confidence.py`)

논문 3.2 공식 그대로:

```
IOCConfidence(o) = 0.30·C_context + 0.25·C_source + 0.20·C_corroboration
                 + 0.15·C_type + 0.10·C_freshness
```

성분 산정 룩업 (모두 0~1):

| 성분 | 규칙 |
|---|---|
| C_context | 작전 키워드 ≥2개=1.0 / 1개=0.7 / 0개=0.3 / 출처 키워드 우세=0.1. curated CSV의 operational=1.0 |
| C_source | 전문기관(Qurium, Mandiant, Viginum, EU DisinfoLab, Meta, Google TAG, EEAS, 정부·CERT)=1.0 / 주요 언론=0.7 / 기타=0.4. curated CSV=1.0 |
| C_corroboration | 독립 출처 1개=0.3 / 2개=0.7 / ≥3개=1.0 |
| C_type | hash·url=1.0 / email=0.9 / domain=0.8 / ns·account=0.6 / ipv4=0.4 / asn=0.3 |
| C_freshness | 사건 기간과 IOC 관측 기간이 겹치면 1.0, 아니면 exp(−gap_days/180) |

합성 IOC 고정값: C_source=0.8, C_corroboration=0.3, 나머지는 규칙대로 (manifest에 기록).

### 4.6 합성 IOC 생성기 (`ioc/synthetic.py`)

**목적**: 실제 IOC가 없는 캠페인의 IOC 계층을 실험 목적상 보강. **연계 판단 근거가 아니라 프레임워크 작동 검증용**임을 코드 주석과 manifest에 명시.

**절대 규칙 — 예약 대역만 사용** (실존 인프라와의 충돌·오귀속 원천 차단):
- 도메인: `.test` TLD만 (RFC 2606). 예: `newsmirror-eu.test`
- IPv4: 문서화 대역만 — 192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24 (RFC 5737)
- ASN: 64496–64511 (RFC 5398)
- 네임서버: `ns1.<random>.test` 형식
- 이메일: `<random>@<random>.test`
- 해시: 랜덤 hex (synthetic=True로 식별)
- 생성 직후 검증 함수로 전수 검사, 위반 시 예외 (테스트 T11)

**생성 알고리즘** (`config/synthetic_scenarios.yaml`로 제어, seed 고정)
1. real IOC가 매칭되지 않은 campaign(크기≥2) 목록 산출
2. 각 campaign에서 사건쌍의 `coverage`(기본 0.65) 비율을 무작위 선택 → 공유 OperationalIOC 부여
   - 유형 배분 `type_mix`: domain 0.40 / url 0.25 / ipv4 0.20 / hash_sha256 0.10 / email 0.05
   - 도메인 부여 시 50% 확률로 공유 NS 관계(`uses_ns`)까지 생성 → I_path 경로 검증용
   - 관측 시점: 각 사건의 first_seen 기준 ±`temporal_jitter_days`(기본 14) 지터
3. **노이즈 주입**: 전체 사건의 `noise_ratio`(기본 0.15) 비율을 캠페인 무관하게 무작위 선택 → 동일한 "공유 호스팅 IP"(문서화 대역)와 "단축 URL 도메인"(.test) 부여. E2가 이 노이즈에 속는지 확인하는 장치.
4. `results/synthetic_manifest.json`에 전체 주입 내역(값, 대상 사건, 규칙, seed) 기록. 동일 seed → manifest의 sha256 동일해야 함 (테스트 T12).

**중요**: coverage < 1.0과 noise_ratio > 0은 config에서 0으로 만들 수 없도록 검증한다 (완전 커버리지 + 무노이즈 = E2 만점 조작 실험이 되어 무의미 — 논문 방어 불가).

---

## 5. 내러티브 축 — N(i,j) (`nlp/embed.py`, `nlp/narrative.py`)

**모델**: sentence-transformers `paraphrase-multilingual-mpnet-base-v2` (config `embedding.model`로 교체 가능). 다국어 지원이므로 언어 감지 불필요.

**전처리**: description에서 URL 제거(내러티브 순수성), 나머지 원문 유지.

**청킹**: 모델 max_seq(512 토큰) 대비 안전하게 — 토크나이저 기준 chunk 220 토큰, overlap 40 토큰. 문단 경계 우선, 초과 시 토큰 단위 분할. 짧은 설명문은 chunk 1개.

**캐시**: `data/processed/embeddings.parquet` (event_id, chunk_id, vector). 텍스트 sha256이 같으면 재계산 금지.

**점수** (논문 3.4.1):
```
N(i,j) = λ·max_sim(i,j) + (1−λ)·avg_topk_sim(i,j)
```
- 두 사건의 모든 chunk 쌍에 대해 코사인 유사도 행렬 → max_sim = 최댓값, avg_topk_sim = 상위 k개 평균 (쌍 수가 k 미만이면 있는 만큼 평균)
- 기본 λ=0.6, k=3
- 코사인은 [−1,1]이므로 `(cos+1)/2`로 [0,1] 정규화 후 결합
- description 결측/공백 사건 → N은 **missing** (0이 아님 — 8장 재정규화로 처리)

**인터페이스**:
```python
def narrative_matrix(events: list[Event], emb: EmbStore, cfg: Config) -> ScoreMatrix:
    """대칭 행렬. 값 ∈ [0,1] 또는 NaN(missing)."""
```

---

## 6. 증거 그래프 & IOC 축 — I(i,j) (`graph/build.py`, `graph/ioc_score.py`)

### 6.1 그래프 구조 (networkx.Graph, 무방향)

노드: `ntype` 속성으로 구분

| ntype | 원천 |
|---|---|
| Event | events.jsonl |
| TTP / Channel | Event 속성 |
| EvidenceSource | Event.evidence_sources |
| IOC | OperationalIOC만 (Benign/EvidenceSourceURL/PlatformContentURL은 그래프 제외) |
| Domain / IP / NS / ASN | IOC 값 분해 + ioc_relations.csv + 합성 관계 |

엣지: `etype`, `weight`, `confidence`, `first_seen`, `last_seen`, `explanation` 속성 필수

| etype | 연결 | confidence 기본 |
|---|---|---|
| LINKED_TO | Event—IOC | IOCConfidence(o) |
| USES / DISTRIBUTED_ON / SUPPORTED_BY | Event—TTP / Channel / EvidenceSource | 1.0 |
| RESOLVES_TO / USES_NS / BELONGS_TO_ASN / REDIRECTS_TO / REGISTERED_AT | 인프라 간 | curated 0.9 / 합성 0.85 |

저장: `data/processed/graph.json` (`nx.node_link_data` — gpickle은 networkx 3.x에서 제거됨).

### 6.2 I_direct — 직접 공유 IOC (논문 3.4.2)

```
I_direct(i,j) = Σ_{o ∈ Oi∩Oj} W(o) / Σ_{o ∈ Oi∪Oj} W(o)
W(o) = type_weight(o) × IOCConfidence(o) × Rarity(o)
```
- type_weight: hash 1.0 / url 0.9 / email 0.8 / domain 0.7 / account·tg_channel 0.6 / ns 0.5 / ipv4 0.3 / asn 0.2
- Rarity(o) = 1/df(o), df = 해당 IOC가 연결된 사건 수 (PoC는 전체 데이터셋 기준; sliding window는 비범위)
- Oi = 사건 i의 OperationalIOC 집합. 한쪽이 공집합이면 I는 **missing**

### 6.3 I_path — 인프라 경로 연결 (논문 3.4.2)

```
I_path(i,j) = max_경로( exp(−ρ·path_length) · path_confidence · temporal_overlap )
temporal_overlap = exp(−gap_days/τ_ioc)
```
- 경로 탐색은 **인프라 서브그래프**(IOC, Domain, IP, NS, ASN 노드)에서만: 사건 i의 IOC 집합 → 사건 j의 IOC 집합 간 최단경로. **다른 Event 노드 경유 금지** (테스트 T-graph). 동일 노드 직접 공유는 I_direct 담당이므로 path_length ≥ 1.
- path_length = 인프라 엣지 수, `max_path_len`(기본 4) 초과 시 0
- path_confidence = 경로 엣지 confidence의 곱
- gap_days = 경로 양끝 IOC의 두 사건별 관측 기간 최소 간격 중 큰 값. 겹치면 0 → overlap=1.0
- τ_ioc는 양끝 IOC 유형 중 짧은 값 적용. 유형별(일): hash 365 / url·email 180 / domain·account 120 / ns 90 / asn 60 / ipv4 30

### 6.4 결합

```
I(i,j) = μ·I_direct + (1−μ)·I_path      # μ=0.6 (논문 미지정 — PoC 기본값, grid 대상)
```

---

## 7. 보조 성분 — D, C, T, A (`scoring/components.py`)

논문 3.4.3의 Jaccard/Overlap 병용:

```
Jaccard(A,B) = |A∩B| / |A∪B|
Overlap(A,B) = |A∩B| / min(|A|,|B|)
```

- **D (TTP)**: 기본 `mix` = 0.5·Jaccard + 0.5·Overlap. config `ttp_sim_mode ∈ {jaccard, overlap, mix}`. 한쪽 TTP 목록이 비면 missing.
- **C (Channel)**: D와 동일 방식, 대상만 channels.
- **T (시간 근접성)**: 두 사건 기간(구간)의 gap_days → `exp(−gap/τ_event)`, τ_event=90일. 겹치면 1.0. 한쪽 날짜 결측 → missing.
- **A (맥락)**: `0.5·actor_match + 0.5·Jaccard(target_countries)`. actor_match = reported_actor 동일 1 / 상이 0. 한쪽 actor 결측이면 target 항만 사용. **평가에서는 사용 금지 (10.3)**.

---

## 8. FCLS 결합 & 결측 재정규화 (`scoring/fcls.py`)

논문 3.4:
```
FCLS(i,j) = αN + βI + γD + δC + εT + ζA
초기값: α=0.30, β=0.30, γ=0.15, δ=0.10, ε=0.10, ζ=0.05
```

결측 재정규화 (논문 3.4.3):
```
FCLS(i,j) = Σ_available(w_k·x_k) / Σ_available(w_k)
```
- missing(NaN) 성분은 분자·분모에서 모두 제외. 0으로 치환 금지.
- 인터페이스: `fcls(components: dict[str,float|nan], weights: dict[str,float]) -> float`

**골든 예시** (테스트 T9로 박제): N=0.8, I=0.5, D=missing, C=0.2, T=1.0, A=제외(ζ=0), 초기 가중치
→ 분자 = 0.30·0.8 + 0.30·0.5 + 0.10·0.2 + 0.10·1.0 = 0.51
→ 분모 = 0.30+0.30+0.10+0.10 = 0.80
→ **FCLS = 0.6375**

---

## 9. Priority(i) — 조사 우선순위 (보조 산출물, `scoring/priority.py`)

논문 3.5. 실험의 주 지표는 FCLS이며 Priority는 표 산출용.

```
Priority(i) = θ1·LinkStrength + θ2·Impact + θ3·EvidenceConfidence
            + θ4·CyberRelevance + θ5·Urgency
결측 시 재정규화 (8장과 동일 방식)
```

| 성분 | 산정 규칙 (0~1) |
|---|---|
| LinkStrength | 해당 사건의 FCLS 상위 3개 평균 (E3 기준) |
| Impact | target_sectors 매핑: 선거·군사·외교 1.0 / 사회갈등·보건 0.7 / 기타 0.5. **판단 불가 시 0.5 중립 대체 + `impact_defaulted=true` 플래그** |
| EvidenceConfidence | min(1, 독립 출처 수/3) × 출처 신뢰 평균(C_source 규칙 재사용) |
| CyberRelevance | OperationalIOC 존재 여부 × 평균 IOCConfidence |
| Urgency | exp(−(오늘−last_seen)/365일). PoC 데이터는 과거 사건이므로 참고치 |

θ 기본값(논문 미지정 — PoC 기본): θ1=0.35, θ2=0.15, θ3=0.15, θ4=0.20, θ5=0.15.
출력: `results/priority_table.csv` (event_id, priority, 성분 분해, 상위 관련 사건 3개, defaulted 플래그).

---

## 10. 실험 설계 (`eval/`)

### 10.1 Ground Truth (`groundtruth.py`)
- **positive pair** = `campaign_id`가 동일한 사건쌍.
- **쿼리 집합 Q** = campaign 크기 ≥ 2인 사건 (positive가 없는 사건은 쿼리에서 제외, 후보 풀에는 포함).
- GT 통계(캠페인 수, 크기 분포, |Q|)를 `results/report.md`에 자동 기록.

### 10.2 과제·지표 (`metrics.py`)
- 과제: **관련 사건 검색** — 각 쿼리 사건 q에 대해 나머지 전 사건을 점수 내림차순 랭킹, 같은 캠페인 사건이 상위에 오는지 평가.
- 지표: P@1, P@3, P@5, MAP, nDCG@10. 직접 구현 + sanity 테스트(완벽 랭킹 → MAP=1.0, T13).
- 참고치: pairwise ROC-AUC (positive/negative 쌍 분류로 본 성능).

### 10.3 실험 조건 (`experiments.py`)

| 조건 | 가중치 | 의미 |
|---|---|---|
| E1 콘텐츠-only | α=1, 나머지 0 | 기존 내러티브 분석 대리 |
| E2 IOC-only | β=1, 나머지 0 | 기존 CTI 매칭 대리 |
| E3 통합 | 초기값에서 **ζ=0** 후 재정규화 | 제안 방식 |
| E3+A (참고) | ζ=0.05 포함 | 순환 위험 민감도 확인용. **본문 결론 근거로 사용 금지** 주석 |

**ζ=0 하드 가드**: `run_evaluation(..., allow_actor=False)` 기본. GT가 campaign 기반(≈reported_actor와 상관)이므로 A항 포함 평가는 순환논리. `allow_actor=True`는 E3+A 전용이며 결과 파일에 `reference_only` 컬럼 표기 (테스트 T10).

### 10.4 Ablation & 가중치 탐색
- **Ablation**: E3에서 N/I/D/C/T를 하나씩 가중치 0으로 (5회) → `results/ablation.csv`.
- **Grid**: α, β ∈ {0.1, 0.2, 0.3, 0.4, 0.5}의 조합, 잔여 가중치는 γ:δ:ε = 1.5:1:1 비율로 분배(합=1). 각 조합의 MAP → `results/gridsearch.csv` + 히트맵. 최고 조합을 report에 명시하되 "소표본 참고치"로 한정.

### 10.5 강건성 실험 (H2 검증 — 논문의 핵심 그림)
- `noise_ratio ∈ {0.0, 0.15, 0.30}` × `coverage ∈ {0.5, 0.65, 0.8}` = 9개 시나리오
- 각 시나리오: 합성 IOC 재생성(동일 seed 체계: base_seed + 시나리오 인덱스) → I 재계산 → E2, E3 재평가
- 출력: `results/robustness.csv` (scenario, noise, coverage, condition, MAP, nDCG) + noise별 MAP 선그래프 (E2는 급락, E3는 완만 — H2)
- 주의: noise=0.0은 강건성 실험 내부에서만 허용 (기본 파이프라인 config에서는 금지 유지)

### 10.6 불확실성
- 쿼리 사건 bootstrap 리샘플 1,000회 → MAP 95% CI → metrics_summary에 CI 컬럼. 소표본(캠페인 수십 개)임을 report 한계 문단에 자동 기재.

---

## 11. 산출물 명세 (파일명 고정)

| 경로 | 내용 |
|---|---|
| `results/pairwise_scores.csv` | i, j, N, I, D, C, T, A, FCLS_E1/E2/E3 (missing은 빈칸) |
| `results/metrics_summary.csv` | condition, P@1, P@3, P@5, MAP, nDCG@10, MAP_CI_low/high |
| `results/ablation.csv` / `gridsearch.csv` / `robustness.csv` | 10.4~10.5 |
| `results/priority_table.csv` | 9장 |
| `results/synthetic_manifest.json` | 4.6 |
| `results/figures/metrics_bar.png` | E1/E2/E3 지표 막대 |
| `results/figures/robustness_lines.png` | noise별 MAP (E2 vs E3) |
| `results/figures/grid_heatmap.png` | α×β MAP 히트맵 |
| `results/evidence_paths/pair_{i}_{j}.html` | FCLS(E3) 상위 3쌍의 pyvis 서브그래프 |
| `results/report.md` | 아래 구성 자동 생성 |

**evidence path 시각화 규칙** (`viz/evidence_path.py`): 두 사건과 이를 잇는 모든 경로(공유 IOC, 인프라 경로, 공유 TTP/Channel)만 추출한 서브그래프. 노드색 = ntype별 고정 팔레트, 엣지 라벨 = etype, IOC 값은 **defang 표기로 출력**. 상위 쌍의 `Event → IOC → NS → IOC → Event` 형태 경로가 눈에 보여야 함 (논문 그림 후보).

**report.md 구성**: ① 실행 환경·폴백 여부 ② 데이터 통계(사건 수, 캠페인 수, real/합성 IOC 수) ③ GT 통계 ④ 조건별 성능표 ⑤ ablation ⑥ 강건성 ⑦ 상위 3쌍 점수 분해 + 증거 경로 파일 링크 ⑧ synthetic manifest 요약 ⑨ 한계 문단 템플릿(소표본, 합성 IOC 의존 범위, ζ 제외 사유).

---

## 12. 마일스톤 & DoD

| M | 범위 | DoD |
|---|---|---|
| M0 | 스캐폴딩, requirements, config 2종 배치 | `pytest` 초록(빈 테스트), `run_all.py --dry-run` 단계 목록 출력 |
| M1 | DISINFOX 클론+로더+스키마 | events.jsonl ≥100건, campaign_id 보유 사건 수 stdout 출력, mapping_report.md 생성 |
| M2 | 임베딩+N | N 행렬 대칭·[0,1]·대각 NaN, 캐시 재실행 시 임베딩 재계산 0건, T5 통과 |
| M3 | IOC 추출·분류·confidence·합성 | T1~T4, T11, T12 통과, iocs.jsonl 생성, manifest 생성 |
| M4 | 그래프+I | graph.json 생성, T6, T7, Event 경유 금지 테스트 통과 |
| M5 | D/C/T/A + FCLS | T8, T9 통과, pairwise_scores.csv 생성 |
| M6 | GT+지표+E1/E2/E3 | metrics_summary.csv 생성, T10, T13 통과 |
| M7 | ablation+grid+강건성 | 3개 csv + 히트맵·선그래프 생성 |
| M8 | 시각화+report | `run_all.py` 클린 환경에서 1회 완주, evidence_paths 3개 html, report.md 완성 |

---

## 13. 테스트 명세 (pytest, 골든 수치 박제)

| ID | 대상 | 입력 → 기대값 |
|---|---|---|
| T1 | refang | `hxxps://bild[.]eu[.]com/x` → `https://bild.eu.com/x` (defang 왕복 일치) |
| T2 | ipv4 검증 | `999.1.1.1` 추출 안 됨, `203.0.113.7` 추출됨 |
| T3 | 분류 | "phishing domain evil-news.test registered by ..." → OperationalIOC / "according to the report at https://qurium.org/..." → EvidenceSourceURL |
| T4 | IOCConfidence | C=(1.0, 1.0, 0.3, 0.8, 1.0) → 0.30+0.25+0.06+0.12+0.10 = **0.83** |
| T5 | N 결합 | max=0.9, avg_top3=0.6, λ=0.6 → **0.78** |
| T6 | I_direct | Oi={d1:0.7, ip1:0.3, u1:0.9}, Oj={d1:0.7, ip2:0.3} → 0.7/2.2 = **0.3182** (±1e-3) |
| T7 | temporal_overlap | gap=30, τ=60 → exp(−0.5) = **0.6065** |
| T8 | Jaccard/Overlap | {t1,t2} vs {t1..t10} → J=**0.2**, O=**1.0**, mix=**0.6** |
| T9 | FCLS 재정규화 | 8장 골든 → **0.6375** |
| T10 | ζ 가드 | allow_actor=False에서 ζ>0 전달 시 예외 발생 |
| T11 | 합성 예약대역 | 생성된 모든 domain이 `.test`, IP가 RFC5737 대역, ASN이 64496–64511 |
| T12 | 합성 결정론 | 동일 seed 2회 생성 → manifest sha256 동일 |
| T13 | 지표 sanity | 완벽 랭킹 → MAP=1.0, nDCG@10=1.0; 역순 랭킹 → MAP < 0.5 |
| T14 | graph 제약 | I_path 경로에 Event 노드 미포함, path_length>4 → 0 |

---

## 14. 비범위 (Out of Scope — 구현 금지)

- 라이브 DNS/WHOIS/인터넷 스캔, 출처 URL 크롤링
- 행위자 귀속 판단·자동 결론 (논문 원칙: 조사 보조)
- GNN/HGT/HAN (Ⅵ장 향후 연구), 웹 대시보드, sliding-window rarity
- 원문 보고서·기사 텍스트의 저장소 내 복제 (URL 참조만)

## 15. 리스크 & 폴백

| 상황 | 폴백 |
|---|---|
| DISINFOX 데이터 파일 미발견 | ① Docker/API로 추출 시도 → ② `scripts/make_fixtures.py`로 샘플 20건 생성(Doppelganger류 6, Ghostwriter류 5, 기타 캠페인 4, 무관 5 — 설명문은 공개 보고서 내용을 **직접 요약해 새로 작성**, 원문 복사 금지) 후 파이프라인 검증. report에 "fixture 모드" 명시 |
| sentence-transformers/torch 설치·다운로드 실패 | config `embedding.backend: tfidf` → sklearn TfidfVectorizer + cosine으로 N 대체. report에 명시 |
| real_iocs.csv 공란 | 정상 동작 (합성만으로 완주). report에 "실제 IOC 0건" 명시 |
| 실행 시간 과다 | 사건 ~120건이면 전수 쌍 ≈ 7,140쌍 — 수 분 내 완료가 정상. 임베딩 캐시 확인 |

---

## 16. `config/weights.yaml` (전문 — 그대로 사용)

```yaml
seed: 42

embedding:
  backend: sbert            # sbert | tfidf(폴백)
  model: paraphrase-multilingual-mpnet-base-v2
  chunk_tokens: 220
  chunk_overlap: 40

narrative:
  lambda: 0.6
  top_k: 3

ioc_score:
  mu: 0.6                   # I_direct 비중 (논문 미지정, PoC 기본)
  rho: 0.5
  max_path_len: 4
  type_weight: {hash_sha256: 1.0, hash_sha1: 1.0, hash_md5: 1.0, url: 0.9,
                email: 0.8, domain: 0.7, account: 0.6, tg_channel: 0.6,
                ns: 0.5, ipv4: 0.3, asn: 0.2}
  tau_days: {hash_sha256: 365, hash_sha1: 365, hash_md5: 365, url: 180,
             email: 180, domain: 120, account: 120, tg_channel: 120,
             ns: 90, asn: 60, ipv4: 30}

components:
  ttp_sim_mode: mix         # jaccard | overlap | mix
  channel_sim_mode: mix
  tau_event_days: 90

fcls:                       # 논문 3.4 초기값
  alpha_N: 0.30
  beta_I: 0.30
  gamma_D: 0.15
  delta_C: 0.10
  epsilon_T: 0.10
  zeta_A: 0.05              # 평가 시 코드가 0으로 강제 (allow_actor=False)

priority:
  theta: {link: 0.35, impact: 0.15, evidence: 0.15, cyber: 0.20, urgency: 0.15}
  impact_default: 0.5

eval:
  p_at: [1, 3, 5]
  ndcg_at: 10
  bootstrap_iters: 1000
  grid_alpha: [0.1, 0.2, 0.3, 0.4, 0.5]
  grid_beta:  [0.1, 0.2, 0.3, 0.4, 0.5]
```

## 17. `config/synthetic_scenarios.yaml` (전문 — 그대로 사용)

```yaml
base_seed: 42
default:
  coverage: 0.65            # 캠페인 내 사건쌍 중 공유 IOC 부여 비율 (1.0 금지)
  noise_ratio: 0.15         # 무관 사건에 저식별력 공유 지표 주입 비율 (기본 파이프라인에서 0.0 금지)
  temporal_jitter_days: 14
  ns_link_prob: 0.5         # 공유 도메인에 NS 관계까지 부여할 확률
  type_mix: {domain: 0.40, url: 0.25, ipv4: 0.20, hash_sha256: 0.10, email: 0.05}

robustness_grid:            # 10.5 — noise=0.0은 이 실험 내부에서만 허용
  noise_ratio: [0.0, 0.15, 0.30]
  coverage: [0.5, 0.65, 0.8]

reserved_only:              # 4.6 절대 규칙 — 검증 함수가 이 목록만 허용
  domain_tld: ".test"
  ipv4_blocks: ["192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24"]
  asn_range: [64496, 64511]
```

---

## 부록 — 사람이 할 일 (Claude Code 범위 밖)

1. `data/curated/real_iocs.csv` 채우기: Qurium Doppelganger 기술 보고서(50+ 도메인 공개), EU DisinfoLab Doppelganger 허브, Mandiant Ghostwriter 보고서에서 도메인·관계를 defang 표기로 옮겨 적기. 비어 있어도 실험은 돌아가지만, 채우면 "실제 IOC 앵커" 서사가 생겨 논문 방어력이 크게 오름.
2. `campaign_map.csv`에 위 캠페인 ↔ DISINFOX 사건 매칭 규칙 작성 (M1의 mapping_report.md를 보고 결정).
3. 결과 검토: report.md의 한계 문단을 논문 Ⅳ장 문체로 다듬기.
