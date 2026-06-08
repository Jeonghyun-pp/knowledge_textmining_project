"""
Streamlit 앱 화면을 자동 캡쳐해 제출용 스크린샷을 생성한다.
- 실행 중인 앱(기본 http://localhost:8601)에 접속
- 예시 쿼리들을 입력 → 추천 실행 → 전체 페이지 PNG + 단계별 카드(①~④,📤) PNG 저장

각 단계는 app.py에서 st.container(border=True) 카드로 감싸져 있어,
그 카드(border wrapper)를 element 스크린샷으로 깔끔하게 캡쳐한다.

사용:
    # 1) 다른 터미널에서 앱 실행
    streamlit run app.py --server.port 8601
    # 2) 캡쳐
    python capture.py
"""
from __future__ import annotations
import os, sys, time
from playwright.sync_api import sync_playwright

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

URL = os.environ.get("APP_URL", "http://localhost:8601")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
os.makedirs(OUT, exist_ok=True)
SCALE = 2

QUERIES = [
    ("q1_school", "오늘 학교 갈 때 입을 깔끔한 꾸안꾸 느낌 추천해줘"),
    ("q2_date_rain", "비 오는 날 데이트룩, 차분하고 단정하게"),
    ("q3_summer", "여름 휴양지에서 시원하고 편한 와이드핏"),
]
# 섹션 헤더 텍스트 → 파일 접미사 (문서 순서대로)
SECTIONS = [
    ("① 자연어 입력", "step1_input"),
    ("② 벡터 임베딩", "step2_embedding"),
    ("③ 추천 과정", "step3_process"),
    ("④ 추천 결과", "step4_result"),
    ("📤 결과 제출", "step5_export"),
]


def card_for(page, heading_text):
    """헤더를 포함하는 bordered container 카드 locator를 반환.

    Streamlit 1.5x의 st.container(border=True)는 헤더의 최근접 stVerticalBlock으로
    렌더되므로, 헤더의 첫 stVerticalBlock 조상을 그 섹션 카드로 사용한다.
    """
    h = page.get_by_role("heading", name=heading_text, exact=False).first
    return h.locator(
        "xpath=ancestor::*[@data-testid='stVerticalBlock'][1]"
    )


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1500, "height": 1000},
                                device_scale_factor=SCALE)
        print(f"[*] open {URL}")
        page.goto(URL, wait_until="networkidle", timeout=120_000)
        page.get_by_text("코퍼스(데이터) 정보", exact=False).first.wait_for(timeout=300_000)
        time.sleep(2)

        for slug, q in QUERIES:
            print(f"[*] query: {q}")
            ta = page.locator("textarea").first
            ta.click()
            ta.fill(q)
            ta.press("Control+Enter")
            time.sleep(1.5)
            page.get_by_role("button", name="추천 실행").first.click()
            page.get_by_role("heading", name="④ 추천 결과", exact=False).first.wait_for(timeout=300_000)
            time.sleep(4)
            # 이미지 lazy-load 대비 끝까지 스크롤 후 복귀
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            time.sleep(2)
            page.evaluate("window.scrollTo(0, 0)")
            time.sleep(1)

            full = os.path.join(OUT, f"{slug}_full.png")
            page.screenshot(path=full, full_page=True)
            print(f"    saved {os.path.basename(full)}")

            for heading, suffix in SECTIONS:
                try:
                    card = card_for(page, heading)
                    card.scroll_into_view_if_needed()
                    time.sleep(0.6)
                    path = os.path.join(OUT, f"{slug}_{suffix}.png")
                    card.screenshot(path=path)
                    print(f"    saved {os.path.basename(path)}")
                except Exception as e:
                    print(f"    [warn] '{heading}' 캡쳐 실패: {type(e).__name__}: {e}")

        browser.close()
        print(f"\n[OK] 스크린샷 저장 폴더: {OUT}")


if __name__ == "__main__":
    main()
