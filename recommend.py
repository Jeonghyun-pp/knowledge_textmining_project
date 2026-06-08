"""
자연어 태그 기반 착장 추천 (기획안 방식)
====================================================
- 임베딩 모델: BGE-M3 (로컬, 무료) — 첫 실행 시 가중치 ~2.2GB 자동 다운로드
- 데이터: dataset.zip 안의 outfits.xlsx (착장 135개 × 7축 태그 = 945개 문장)
- 스코어링: 착장별 상위 3개 축 cosine 평균 (top-3 mean)

오프라인(1회) : 945개 태그 임베딩 → embeddings/ 에 캐시
온라인(매 쿼리): 쿼리 1개 임베딩 → cosine → 착장별 top-3 mean → top-K

데이터는 압축 해제 없이 zip에서 직접 읽는다.
"""
from __future__ import annotations

import io
import os
import json
import zipfile
from dataclasses import dataclass
from typing import Optional

import numpy as np

# ── 설정 ───────────────────────────────────────────────────────────
AXES = ["item", "color", "style", "mood", "occasion", "season_weather", "fit_silhouette"]
N_AXES = len(AXES)

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_default_zip() -> str:
    """dataset.zip 위치를 자동 탐색 (모노레포/독립레포 양쪽 지원).

    우선순위:
      1) ../data/dataset.zip   (모노레포: outfit_recommender 의 상위 data/)
      2) ./data/dataset.zip    (독립레포: repo_root/data/)
      3) ./dataset.zip         (repo_root 바로 아래)
    셋 다 없으면 안내용으로 2)번 경로를 반환한다.
    """
    candidates = [
        os.path.join(_HERE, "..", "data", "dataset.zip"),
        os.path.join(_HERE, "data", "dataset.zip"),
        os.path.join(_HERE, "dataset.zip"),
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c
    return os.path.normpath(candidates[1])


DEFAULT_ZIP = _resolve_default_zip()
CACHE_DIR = os.path.join(_HERE, "embeddings")
VEC_PATH = os.path.join(CACHE_DIR, "tag_vectors.npy")   # shape (n_outfits, 7, dim)
META_PATH = os.path.join(CACHE_DIR, "tag_meta.json")

MODEL_NAME = "BAAI/bge-m3"

# 모듈 1회 로딩 캐시
_model = None
_cache = None  # (vectors, meta)


# ── 데이터 로딩 (zip에서 직접) ──────────────────────────────────────
def _find_member(zf: zipfile.ZipFile, suffix: str) -> Optional[str]:
    """zip 내부에서 경로 끝이 suffix와 일치하는 첫 엔트리를 찾는다."""
    suffix = suffix.replace("\\", "/").lstrip("/")
    for name in zf.namelist():
        if name.replace("\\", "/").endswith(suffix):
            return name
    return None


def load_outfits(zip_path: str = DEFAULT_ZIP) -> list[dict]:
    """
    outfits.xlsx를 읽어 착장 리스트를 반환.
    각 원소: {image_id, image_path, tags: {axis: text, ...}}
    """
    import openpyxl

    with zipfile.ZipFile(zip_path) as zf:
        xlsx_name = _find_member(zf, "outfits.xlsx")
        if xlsx_name is None:
            raise FileNotFoundError("outfits.xlsx를 zip에서 찾을 수 없습니다.")
        with zf.open(xlsx_name) as f:
            data = f.read()

    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else "" for h in rows[0]]
    idx = {h: i for i, h in enumerate(header)}

    missing = [a for a in AXES if a not in idx]
    if missing:
        raise ValueError(f"xlsx에 없는 태그 축: {missing} (실제 컬럼: {header})")

    outfits = []
    for r in rows[1:]:
        if r is None or r[idx["image_id"]] in (None, ""):
            continue
        tags = {a: (str(r[idx[a]]).strip() if r[idx[a]] is not None else "") for a in AXES}
        # 7축 중 하나라도 비면 스킵 (현재 데이터는 결측 0건)
        if any(not tags[a] for a in AXES):
            continue
        outfits.append({
            "image_id": str(r[idx["image_id"]]).strip(),
            "image_path": str(r[idx["image_path"]]).strip() if "image_path" in idx and r[idx["image_path"]] else "",
            "tags": tags,
        })
    return outfits


def load_image(image_id: str, zip_path: str = DEFAULT_ZIP):
    """zip에서 image_id에 해당하는 이미지를 PIL.Image로 반환 (시연용)."""
    from PIL import Image

    with zipfile.ZipFile(zip_path) as zf:
        # image_id 기준으로 images/ 안에서 매칭
        member = None
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".PNG"):
            member = _find_member(zf, f"images/{image_id}{ext}")
            if member:
                break
        if member is None:
            raise FileNotFoundError(f"이미지 없음: {image_id}")
        with zf.open(member) as f:
            return Image.open(io.BytesIO(f.read())).convert("RGB")


# ── 임베딩 모델 ────────────────────────────────────────────────────
def get_model():
    """BGE-M3 모델 로드 (1회). 첫 실행 시 가중치 다운로드(~2.2GB)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _encode(texts: list[str]) -> np.ndarray:
    """텍스트 → L2 정규화 임베딩 (cosine = 내적이 되도록)."""
    model = get_model()
    vecs = model.encode(
        texts,
        normalize_embeddings=True,   # ← cosine 위해 필수
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


# ── 오프라인: 임베딩 빌드/로드 ─────────────────────────────────────
def build_or_load_embeddings(zip_path: str = DEFAULT_ZIP, rebuild: bool = False):
    """
    945개 태그 임베딩을 빌드하거나 캐시에서 로드.
    returns: (vectors[n,7,dim], meta) — meta = {"axes", "outfits":[{image_id,image_path}], "model"}
    """
    global _cache
    if _cache is not None and not rebuild:
        return _cache

    if (not rebuild) and os.path.exists(VEC_PATH) and os.path.exists(META_PATH):
        vectors = np.load(VEC_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        _cache = (vectors, meta)
        return _cache

    # 빌드
    outfits = load_outfits(zip_path)
    # 평탄화: outfit0[axis0..6], outfit1[...], ... 순서 보장
    texts = [o["tags"][a] for o in outfits for a in AXES]
    flat = _encode(texts)                       # (n*7, dim)
    dim = flat.shape[1]
    vectors = flat.reshape(len(outfits), N_AXES, dim)

    meta = {
        "model": MODEL_NAME,
        "axes": AXES,
        "outfits": [{"image_id": o["image_id"], "image_path": o["image_path"]} for o in outfits],
        # 설명/검수용으로 원문 태그도 함께 저장
        "tags": [o["tags"] for o in outfits],
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(VEC_PATH, vectors)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    _cache = (vectors, meta)
    return _cache


# ── 온라인: 추천 ───────────────────────────────────────────────────
@dataclass
class OutfitMatch:
    rank: int
    image_id: str
    image_path: str
    score: float                       # top-3 축 cosine 평균
    matched_axes: list[tuple[str, float]]  # 추천 근거: 상위 축과 점수


@dataclass
class RecommendTrace:
    """추천 1회의 전 과정을 담은 추적 결과 (시각화/디버깅/제출용)."""
    query: str
    query_vector: np.ndarray           # (dim,) — L2 정규화된 쿼리 임베딩
    sims: np.ndarray                   # (n, 7) — 착장×축 cosine 전체 행렬
    scores: np.ndarray                 # (n,)  — 착장별 top-3 평균 점수
    topk_idx: np.ndarray               # (n, top_axes) — 착장별 선택된 상위 축 인덱스
    matches: list[OutfitMatch]         # 최종 top-K 추천
    meta: dict                         # axes/outfits/tags/model 메타


def recommend_with_trace(query: str, top_k: int = 5, top_axes: int = 3,
                         zip_path: str = DEFAULT_ZIP) -> RecommendTrace:
    """
    자연어 쿼리 → 착장 top-K 추천 + 중간 계산값 전체를 함께 반환.
    score_i = mean( 착장 i의 7개 축 cosine 중 상위 top_axes개 )
    """
    if not query or len(query.strip()) < 2:
        raise ValueError("쿼리는 2글자 이상 입력하세요.")

    vectors, meta = build_or_load_embeddings(zip_path)
    n, a, dim = vectors.shape

    qv = _encode([query])[0]                     # (dim,) 정규화됨
    sims = vectors @ qv                          # (n, 7) — cosine (둘 다 정규화)

    k = min(top_axes, a)
    topk_idx = np.argsort(-sims, axis=1)[:, :k]  # (n, k) — 각 착장 상위 k축
    rows = np.arange(n)[:, None]
    top_sims = sims[rows, topk_idx]              # (n, k)
    scores = top_sims.mean(axis=1)               # (n,)

    order = np.argsort(-scores)[:top_k]
    matches = []
    for rank, i in enumerate(order, 1):
        axes_order = topk_idx[i]
        matched = [(AXES[j], float(sims[i, j])) for j in axes_order]
        matches.append(OutfitMatch(
            rank=rank,
            image_id=meta["outfits"][i]["image_id"],
            image_path=meta["outfits"][i]["image_path"],
            score=float(scores[i]),
            matched_axes=matched,
        ))

    return RecommendTrace(
        query=query, query_vector=qv, sims=sims, scores=scores,
        topk_idx=topk_idx, matches=matches, meta=meta,
    )


def recommend(query: str, top_k: int = 5, top_axes: int = 3,
              zip_path: str = DEFAULT_ZIP) -> list[OutfitMatch]:
    """
    자연어 쿼리 → 착장 top-K 추천.
    score_i = mean( 착장 i의 7개 축 cosine 중 상위 top_axes개 )
    """
    return recommend_with_trace(query, top_k=top_k, top_axes=top_axes,
                                zip_path=zip_path).matches


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "오늘 학교 갈 때 입을 깔끔한 꾸안꾸 느낌 추천해줘"
    print(f"쿼리: {q}\n")
    for m in recommend(q, top_k=5):
        axes = ", ".join(f"{name}={s:.3f}" for name, s in m.matched_axes)
        print(f"{m.rank}. {m.image_id}  score={m.score:.4f}  [{axes}]")
