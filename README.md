# 자연어 태그 기반 착장 추천

기획안 방식(자연어 태그 ↔ 사용자 입력 의미 유사도)을 그대로 구현한 추천 모듈.

- **임베딩**: BGE-M3 (로컬, 무료)
- **데이터**: `../data/dataset.zip` 안의 `outfits.xlsx` (착장 135개 × 7축 태그)
- **스코어링**: 착장별 상위 3개 축 cosine 평균 (**top-3 mean**)

## 동작 구조
```
[오프라인 1회]  outfits.xlsx 7축 태그 945개 → BGE-M3 인코딩 → embeddings/ 캐시
[온라인 매쿼리] 쿼리 1개 → BGE-M3 인코딩 → cosine(945) → 착장별 top-3 mean → top-K
```
데이터는 **압축 해제 불필요** — `dataset.zip`에서 직접 읽습니다.

## 팀원 셋업 (GitHub clone 후)
> ⚠️ `data/dataset.zip`(≈158MB, 착장 이미지 + `outfits.xlsx`)은 GitHub 100MB 제한을 넘어 **레포에 포함되지 않습니다.**
> 아래 **0번**을 먼저 수행해야 ④ 결과의 착장 이미지가 표시됩니다.

**0) 데이터셋 내려받기** → 📥 [Google Drive에서 `dataset.zip` 받기](https://drive.google.com/file/d/1AIjPgX9UwJtAMM0Whg9Z3EEE8JmTxT_v/view?usp=sharing)

받은 `dataset.zip`을 **압축 해제하지 말고** 레포의 `data/` 폴더에 그대로 둡니다.
최종 경로 → `<repo>/data/dataset.zip`

```bash
# 0) 데이터셋 내려받기
#  (A) 위 Google Drive 링크에서 직접 받아 data/dataset.zip 로 저장, 또는
#  (B) 터미널에서 gdown 으로 바로 받기 ↓
pip install gdown
gdown 1AIjPgX9UwJtAMM0Whg9Z3EEE8JmTxT_v -O data/dataset.zip

# 1) 의존성 설치
pip install -r requirements.txt

# 2) 실행
python -m streamlit run app.py
```
- **BGE-M3 모델(~2.2GB)** 은 첫 실행 시 자동 다운로드됩니다(인터넷 1회 필요).
- **임베딩 캐시(`embeddings/`)** 는 레포에 포함되어 있어 재빌드 없이 ②③ 단계가 바로 동작합니다.
  (만약 없으면 `dataset.zip`만 있으면 첫 실행 때 자동 생성됩니다.)
- `streamlit` 명령이 PATH에 없으면 위처럼 **`python -m streamlit`** 로 실행하세요.

## 실행 방법
```bash
pip install -r requirements.txt

# ★ Streamlit 데모/제출 앱 (권장) — 입력·임베딩·과정·결과를 한 화면에서 시각화
python -m streamlit run app.py

# 데모 노트북
jupyter notebook demo.ipynb

# 또는 CLI
python recommend.py "오늘 학교 갈 때 입을 깔끔한 꾸안꾸 느낌 추천해줘"
```

### Streamlit 앱 화면 구성 (제출용)
`streamlit run app.py` 를 실행하면 아래 4단계가 순서대로 표시되어 그대로 캡쳐할 수 있습니다.

| 단계 | 내용 |
|---|---|
| ① 자연어 입력 | 텍스트 입력 + 예시 버튼 |
| ② 벡터 임베딩 | 쿼리 임베딩(1024d, L2정규화) 분포 + 임베딩 공간 **PCA 2D 투영**(쿼리·추천 강조) |
| ③ 추천 과정 | 착장×7축 **cosine 히트맵 표** + top-3 평균 점수 산정(반영 축 강조) |
| ④ 추천 결과 | top-K 착장 이미지 카드 + 점수 + 근거 축 |
| 📤 결과 제출 | 결과 **JSON** / 과정 **CSV** 다운로드, `runs/` 폴더 저장 |

### 제출용 스크린샷 자동 생성
앱을 띄운 뒤 `capture.py`를 실행하면 예시 쿼리 3개에 대해 **전체 페이지 + 단계별 카드(①~④,📤)**
스크린샷을 `screenshots/` 에 자동 저장합니다.
```bash
# 터미널 1
python -m streamlit run app.py --server.port 8601
# 터미널 2
python capture.py            # → screenshots/{쿼리}_{단계}.png
```

## 산출물 폴더
| 폴더 | 내용 |
|---|---|
| `screenshots/` | 제출용 캡쳐 (쿼리별 full + 단계별 PNG) |
| `runs/` | 추천 결과 JSON (쿼리·임베딩 미리보기·점수·근거 축·태그 포함) |

> ⚠️ **첫 실행 시** BGE-M3 가중치(~2.2GB)가 HuggingFace에서 자동 다운로드됩니다(인터넷 1회 필요).
> 이후에는 캐시되어 오프라인에서도 동작합니다.
> 임베딩 캐시(`embeddings/`)도 첫 빌드 후 재사용되어, 두 번째 실행부터는 모델 추론만 합니다.

## 파일
| 파일 | 역할 |
|---|---|
| `recommend.py` | 데이터 로딩 · 임베딩 빌드/캐시 · 추천 함수 (`recommend(query)`) |
| `demo.ipynb` | 시연용 — 쿼리 입력 → 착장 이미지 top-K 표시 |
| `embeddings/` | 태그 임베딩 캐시 (`tag_vectors.npy`, `tag_meta.json`) — 자동 생성 |
| `requirements.txt` | 의존성 |

## 추천 근거(설명 가능성)
각 추천은 `matched_axes`로 **어느 축(item/color/style/mood/occasion/season_weather/fit_silhouette)이
가장 강하게 매칭됐는지**를 함께 반환합니다.
