"""
자연어 태그 기반 착장 추천 — Streamlit 데모/제출용 앱
====================================================
한 화면에서 ① 자연어 입력 ② 벡터 임베딩 ③ 추천 과정 ④ 추천 결과 를 모두
시각화하고, 결과를 JSON/CSV로 내보내(제출) 수 있게 한다.

실행:
    cd outfit_recommender
    streamlit run app.py
"""
from __future__ import annotations

import io
import os
import json
import datetime as dt

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import streamlit as st

import recommend as R

# ── 페이지 설정 ────────────────────────────────────────────────────
st.set_page_config(page_title="OOTD AI — 자연어 착장 추천", page_icon="👕", layout="wide")

RUNS_DIR = os.path.join(R._HERE, "runs")
AXES_KO = {
    "item": "아이템", "color": "색상", "style": "스타일", "mood": "무드",
    "occasion": "상황", "season_weather": "계절/날씨", "fit_silhouette": "핏/실루엣",
}
EXAMPLES = [
    "오늘 학교 갈 때 입을 깔끔한 꾸안꾸 느낌 추천해줘",
    "비 오는 날 데이트룩, 차분하고 단정하게",
    "여름 휴양지에서 시원하고 편한 와이드핏",
    "면접에 입을 단정한 정장 느낌",
    "힙하고 편한 스트릿 캐주얼 데일리룩",
    "한겨울 추운 날 따뜻하게 껴입는 코디",
]


# ── 캐시 자원 ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="BGE-M3 임베딩 로딩 중… (최초 1회 모델 다운로드)")
def load_engine():
    """임베딩 캐시(또는 빌드)와 모델을 1회 로드."""
    vectors, meta = R.build_or_load_embeddings()
    R.get_model()  # 모델 워밍업 (쿼리 인코딩 대비)
    return vectors, meta


@st.cache_data(show_spinner=False)
def pca_2d(_vectors_hash: str):
    """착장 평균 벡터(135×1024)를 2D PCA로 투영. 캐시 키는 해시 문자열."""
    from sklearn.decomposition import PCA
    vectors, _ = load_engine()
    outfit_mean = vectors.mean(axis=1)            # (n, dim) 착장당 7축 평균
    outfit_mean = outfit_mean / (np.linalg.norm(outfit_mean, axis=1, keepdims=True) + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    pts = pca.fit_transform(outfit_mean)          # (n, 2)
    return pts, pca, outfit_mean


@st.cache_data(show_spinner=False)
def load_image_bytes(image_id: str):
    """zip에서 이미지 바이트를 읽어 캐시 (st.image 재호출 비용 절감)."""
    try:
        img = R.load_image(image_id)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


# ── 헤더 ───────────────────────────────────────────────────────────
st.title("👕 OOTD AI — 자연어 기반 착장 추천")
st.caption(
    "BGE-M3 임베딩으로 자연어 입력과 착장 7축 태그의 의미 유사도를 계산해 추천합니다. "
    "**① 자연어 입력 → ② 벡터 임베딩 → ③ 추천 과정 → ④ 결과** 순서로 보여줍니다."
)

with st.sidebar:
    st.header("⚙️ 설정")
    top_k = st.slider("추천 개수 (top-K)", 1, 10, 5)
    top_axes = st.slider("점수 산정 축 수 (top-N mean)", 1, 7, 3,
                         help="착장의 7개 축 cosine 중 상위 N개의 평균을 최종 점수로 사용")
    show_pca = st.checkbox("임베딩 공간 2D 시각화(PCA)", value=True)
    st.divider()
    vectors, meta = load_engine()
    st.markdown("**📦 코퍼스(데이터) 정보**")
    st.markdown(
        f"- 임베딩 모델: `{meta['model']}`\n"
        f"- 착장 수: **{vectors.shape[0]}**개\n"
        f"- 태그 축: **{vectors.shape[1]}**개 {list(AXES_KO.values())}\n"
        f"- 임베딩 차원: **{vectors.shape[2]}**d\n"
        f"- 총 태그 벡터: **{vectors.shape[0] * vectors.shape[1]}**개"
    )


# ── ① 자연어 입력 ──────────────────────────────────────────────────
if "query" not in st.session_state:
    st.session_state.query = EXAMPLES[0]

with st.container(border=True):
    st.subheader("① 자연어 입력")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.query = ex

    query = st.text_area("원하는 분위기·상황을 자연어로 입력하세요", key="query", height=80)
    run = st.button("🔍 추천 실행", type="primary", use_container_width=True)

if not run:
    st.info("예시 버튼을 누르거나 직접 입력한 뒤 **추천 실행**을 누르세요.")
    st.stop()

if not query or len(query.strip()) < 2:
    st.error("쿼리는 2글자 이상 입력하세요.")
    st.stop()

# ── 추천 실행 (전 과정 추적) ───────────────────────────────────────
with st.spinner("쿼리 임베딩 + 유사도 계산 중…"):
    trace = R.recommend_with_trace(query, top_k=top_k, top_axes=top_axes)

axes = trace.meta["axes"]
n_outfits = trace.sims.shape[0]


# ── ② 벡터 임베딩 ──────────────────────────────────────────────────
with st.container(border=True):
    st.subheader("② 벡터 임베딩")
    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown(f"**쿼리 임베딩 벡터** — `{trace.query_vector.shape[0]}`차원, "
                    f"L2 norm = `{np.linalg.norm(trace.query_vector):.4f}` (정규화됨)")
        head = trace.query_vector[:48]
        df_vec = pd.DataFrame({"dim": np.arange(head.shape[0]), "value": head})
        st.bar_chart(df_vec, x="dim", y="value", height=200)
        st.caption("쿼리 벡터의 앞 48개 차원 (전체 1024d 중 일부). 코사인 = 정규화 벡터 내적.")

    with c2:
        if show_pca:
            pts, _pca, _ = pca_2d("v1")
            df_pca = pd.DataFrame(pts, columns=["x", "y"])
            df_pca["종류"] = "착장 (전체)"
            df_pca["label"] = [o["image_id"] for o in trace.meta["outfits"]]
            # top-K 강조
            topk_ids = {m.image_id for m in trace.matches}
            df_pca.loc[df_pca["label"].isin(topk_ids), "종류"] = "추천 top-K"
            # 쿼리 투영
            q2d = _pca.transform(trace.query_vector[None, :])[0]
            df_q = pd.DataFrame({"x": [q2d[0]], "y": [q2d[1]], "종류": ["쿼리"], "label": ["query"]})
            df_plot = pd.concat([df_pca, df_q], ignore_index=True)
            st.scatter_chart(df_plot, x="x", y="y", color="종류", height=300)
            st.caption("임베딩 공간 PCA 2D 투영 — 쿼리(빨강)와 가까운 착장이 추천됩니다.")
        else:
            st.caption("사이드바에서 PCA 시각화를 켜면 임베딩 공간이 표시됩니다.")


# ── ③ 추천 과정 ────────────────────────────────────────────────────
# 상위 후보(추천된 top-K)에 대한 축별 cosine 표 (export에서도 재사용)
match_order = [next(i for i, o in enumerate(trace.meta["outfits"]) if o["image_id"] == m.image_id)
               for m in trace.matches]
rows = []
for m, idx in zip(trace.matches, match_order):
    chosen = set(int(j) for j in trace.topk_idx[idx])
    row = {"rank": m.rank, "image_id": m.image_id}
    for j, ax in enumerate(axes):
        row[AXES_KO[ax]] = trace.sims[idx, j]
    row["최종점수"] = m.score
    rows.append((row, chosen, axes))

df_proc = pd.DataFrame([r[0] for r in rows]).set_index("rank")


def _style(df):
    sty = df.style.format({**{AXES_KO[a]: "{:.3f}" for a in axes}, "최종점수": "{:.4f}"})
    sty = sty.background_gradient(cmap="Greens", subset=[AXES_KO[a] for a in axes], vmin=0, vmax=0.8)
    sty = sty.background_gradient(cmap="Blues", subset=["최종점수"])
    # 선택된 축 굵게
    def bold_chosen(row):
        r_idx = list(df.index).index(row.name)
        chosen, ax_list = rows[r_idx][1], rows[r_idx][2]
        styles = []
        for col in df.columns:
            ko2ax = {AXES_KO[a]: a for a in ax_list}
            if col in ko2ax and ax_list.index(ko2ax[col]) in chosen:
                styles.append("font-weight:700; border:2px solid #2e7d32")
            else:
                styles.append("")
        return styles
    return sty.apply(bold_chosen, axis=1)


with st.container(border=True):
    st.subheader("③ 추천 과정 — 축별 코사인 유사도 → top-N 평균")
    st.markdown(
        f"각 착장의 **7개 축** 각각에 대해 쿼리와의 cosine을 계산하고, "
        f"그중 **상위 {top_axes}개 축의 평균**을 최종 점수로 사용합니다 (선택된 축은 **굵게**)."
    )
    st.dataframe(_style(df_proc), use_container_width=True)
    st.caption("초록 = 축별 cosine(진할수록 높음) · 굵은 테두리 = 점수에 반영된 상위 축 · 파랑 = 최종 점수")


# ── ④ 추천 결과 ────────────────────────────────────────────────────
tags_all = trace.meta.get("tags", None)
with st.container(border=True):
    st.subheader("④ 추천 결과")
    res_cols = st.columns(min(len(trace.matches), 5))
    for i, m in enumerate(trace.matches):
        col = res_cols[i % len(res_cols)]
        with col:
            img = load_image_bytes(m.image_id)
            if img:
                col.image(img, use_container_width=True)
            else:
                col.warning(f"이미지 없음\n{m.image_id}")
            col.markdown(f"**{m.rank}위 · {m.image_id}**")
            col.markdown(f"점수 `{m.score:.4f}`")
            badges = " ".join(f"`{AXES_KO[a]} {s:.2f}`" for a, s in m.matched_axes)
            col.markdown(f"근거: {badges}")


# ── 결과 제출(내보내기) ────────────────────────────────────────────
now = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
result_payload = {
    "query": trace.query,
    "model": trace.meta["model"],
    "params": {"top_k": top_k, "top_axes": top_axes},
    "corpus": {"n_outfits": int(n_outfits), "n_axes": len(axes), "dim": int(trace.query_vector.shape[0])},
    "query_vector_preview": trace.query_vector[:16].round(5).tolist(),
    "timestamp": now,
    "results": [
        {
            "rank": m.rank,
            "image_id": m.image_id,
            "image_path": m.image_path,
            "score": round(m.score, 6),
            "matched_axes": [{"axis": a, "cosine": round(s, 6)} for a, s in m.matched_axes],
            "tags": tags_all[match_order[i]] if tags_all else None,
        }
        for i, m in enumerate(trace.matches)
    ],
}
json_bytes = json.dumps(result_payload, ensure_ascii=False, indent=2).encode("utf-8")

df_csv = df_proc.reset_index()
csv_bytes = df_csv.to_csv(index=False).encode("utf-8-sig")

with st.container(border=True):
    st.subheader("📤 결과 제출 / 내보내기")
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ 결과 JSON", json_bytes, file_name=f"recommend_{now}.json",
                       mime="application/json", use_container_width=True)
    c2.download_button("⬇️ 과정 CSV (축별 cosine)", csv_bytes, file_name=f"process_{now}.csv",
                       mime="text/csv", use_container_width=True)
    if c3.button("💾 runs/ 폴더에 저장", use_container_width=True):
        os.makedirs(RUNS_DIR, exist_ok=True)
        path = os.path.join(RUNS_DIR, f"recommend_{now}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(json_bytes.decode("utf-8"))
        st.success(f"저장 완료: `{path}`")

    with st.expander("제출용 JSON 미리보기"):
        st.json(result_payload)
