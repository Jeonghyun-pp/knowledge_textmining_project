"""
E2E + 구분력(discrimination) 검증 테스트
- 여러 상반된 쿼리에 대해 top-K가 실제로 갈리는지 정량 측정
- 1) 실행 확인  2) 교차 겹침(Jaccard)  3) 특정 착장 독점 여부
  4) 점수 분리도  5) 상반 쿼리 대비(semantic sanity)
"""
import os, sys, json
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import recommend as R

# 의도적으로 서로 다른 축을 자극하는 쿼리들
QUERIES = {
    "여름_시원": "한여름 더운 날 시원하고 가벼운 옷",
    "겨울_따뜻": "한겨울 추운 날 따뜻하게 껴입는 코디",
    "포멀_정장": "면접이나 발표에 입을 단정한 정장 느낌",
    "스트릿_캐주얼": "힙하고 편한 스트릿 캐주얼 데일리룩",
    "데이트_로맨틱": "비 오는 날 데이트, 차분하고 로맨틱하게",
    "운동_액티브": "활동적인 운동이나 등산에 편한 옷",
}
TOP_K = 5

def main():
    vectors, meta = R.build_or_load_embeddings()
    n = vectors.shape[0]
    print(f"[SETUP] 착장 {vectors.shape[0]} × 축 {vectors.shape[1]} × {vectors.shape[2]}d 로드\n")

    results = {}
    all_scores = {}
    for name, q in QUERIES.items():
        recs = R.recommend(q, top_k=TOP_K)
        results[name] = [m.image_id for m in recs]
        # 전체 점수 분포도 보기 위해 raw score 재계산
        qv = R._encode([q])[0]
        sims = vectors @ qv
        k = min(3, vectors.shape[1])
        topk = np.sort(sims, axis=1)[:, -k:]
        scores = topk.mean(axis=1)
        all_scores[name] = np.sort(scores)[::-1]
        top1 = recs[0]
        print(f"[{name}] \"{q}\"")
        for m in recs:
            axes = ", ".join(f"{a}({s:.2f})" for a, s in m.matched_axes)
            print(f"    {m.rank}. {m.image_id}  score={m.score:.3f}  근거:{axes}")
        print()

    # ── 1) 교차 겹침 (Jaccard) ────────────────────────────
    print("=" * 60)
    print("1) 쿼리 간 top-5 겹침 (Jaccard, 낮을수록 구분 잘됨)")
    names = list(QUERIES.keys())
    jaccs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = set(results[names[i]]), set(results[names[j]])
            jac = len(a & b) / len(a | b)
            jaccs.append(jac)
            mark = "  <- 겹침!" if jac >= 0.4 else ""
            print(f"    {names[i]:14s} vs {names[j]:14s}: {jac:.2f}{mark}")
    print(f"    >> 평균 Jaccard = {np.mean(jaccs):.3f}  (0=완전분리, 1=동일)")

    # ── 2) 특정 착장 독점 여부 ────────────────────────────
    print("\n2) 착장 출현 빈도 (특정 착장이 모든 쿼리 독점하는지)")
    from collections import Counter
    cnt = Counter(iid for lst in results.values() for iid in lst)
    total_slots = len(QUERIES) * TOP_K
    uniq = len(cnt)
    print(f"    총 {total_slots}슬롯 중 고유 착장 {uniq}개 (다양성 {uniq/total_slots:.0%})")
    dom = [f"{iid}×{c}" for iid, c in cnt.most_common(5) if c > 1]
    print(f"    2회 이상 등장: {dom if dom else '없음(완전 분리)'}")

    # ── 3) 점수 분리도 ────────────────────────────────────
    print("\n3) 점수 분리도 (top1 vs 5위, 클수록 확신 강함)")
    for name in names:
        s = all_scores[name]
        margin = s[0] - s[4]
        print(f"    {name:14s}: top1={s[0]:.3f}  5위={s[4]:.3f}  margin={margin:.3f}  "
              f"전체평균={s.mean():.3f}")

    # ── 4) 상반 쿼리 semantic sanity ──────────────────────
    print("\n4) 상반 쿼리 sanity (여름 vs 겨울 결과가 안 겹쳐야 정상)")
    summer, winter = set(results["여름_시원"]), set(results["겨울_따뜻"])
    overlap = summer & winter
    print(f"    여름∩겨울 = {overlap if overlap else '∅ (완벽 분리 ✓)'}")
    formal, street = set(results["포멀_정장"]), set(results["스트릿_캐주얼"])
    print(f"    정장∩스트릿 = {formal & street if (formal&street) else '∅ (완벽 분리 ✓)'}")

    # ── 종합 판정 ─────────────────────────────────────────
    print("\n" + "=" * 60)
    avg_jac = np.mean(jaccs)
    div = uniq / total_slots
    verdict = []
    verdict.append("PASS" if avg_jac < 0.25 else ("WARN" if avg_jac < 0.45 else "FAIL"))
    verdict.append("PASS" if div > 0.7 else ("WARN" if div > 0.5 else "FAIL"))
    print(f"종합: 겹침={avg_jac:.2f}[{verdict[0]}]  다양성={div:.0%}[{verdict[1]}]")

if __name__ == "__main__":
    main()
