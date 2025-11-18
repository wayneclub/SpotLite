import re
import urllib.parse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

# ========= 共用：建立英文介面的 Chrome =========


def create_en_browser(headless: bool = False,
                      window_size: str = "1280,900") -> webdriver.Chrome:
    """
    建立一個預設就是英文 UI 的 Chrome WebDriver。
    reviews / details 都可以共用。
    """
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={window_size}")
    options.add_argument("--lang=en-US")
    options.add_experimental_option(
        "prefs", {"intl.accept_languages": "en,en_US"}
    )
    # 你原本在 maps_details 裡的 user-agent 可以一起搬過來：
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        " AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    )

    browser = webdriver.Chrome(options=options)

    # Best-effort: force Accept-Language headers
    try:
        browser.execute_cdp_cmd("Network.enable", {})
        browser.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
            "headers": {"Accept-Language": "en-US,en;q=0.9"}
        })
    except Exception:
        pass

    return browser


def ensure_hl_en(browser):
    """
    確保目前頁面的 URL 帶有 hl=en&gl=US。
    （就是你 maps_reivews.py 頂端那段 hl=en 補丁）
    """
    try:
        cur = browser.current_url
        if "hl=" not in cur:
            sep = "&" if "?" in cur else "?"
            browser.get(cur + f"{sep}hl=en&gl=US")
    except Exception:
        pass


# ========= 共用：展開分享短鏈、抽出 place_id / text_query =========

def resolve_with_selenium(url: str) -> str:
    """
    這裡請你把 maps_details.py 目前的 resolve_with_selenium 內容整段剪下貼進來，
    只需要把裡面的 webdriver.Chrome(...) 換成 create_en_browser(headless=True) 即可。
    """
    browser = create_en_browser(headless=True, window_size="1280,800")
    try:
        browser.set_page_load_timeout(25)
        browser.get(url)
        WebDriverWait(browser, 15).until(
            lambda d: "google.com/maps" in d.current_url
                      or "google.co" in d.current_url
        )
        return browser.current_url
    except Exception:
        return url
    finally:
        try:
            browser.quit()
        except WebDriverException:
            pass


def expand_maps_share_url(url: str) -> str:
    """
    專門處理 maps.app.goo.gl / goo.gl/maps 等短鏈。
    你原本 maps_details.py 的 expand_maps_share_url 可以直接搬過來，
    我這邊先給一個簡化版。
    """
    try:
        if url.startswith("https://maps.app.goo.gl") or url.startswith("http://maps.app.goo.gl"):
            return resolve_with_selenium(url) or url
        if url.startswith("https://goo.gl/maps") or url.startswith("http://goo.gl/maps"):
            return resolve_with_selenium(url) or url
        # 如果已經是 /maps/place 就直接回傳
        if "google.com/maps/place/" in url or "google.co.jp/maps/place/" in url:
            return url
        return url
    except Exception:
        return url


def extract_place_info(url: str):
    """
    從 Google Maps URL 抽出 place_id 或 text_query。
    你 maps_details.py 原本的 extract_place_info 也可以直接搬進來，
    這裡是照你現有邏輯改成共用版。
    """
    expanded = expand_maps_share_url(url)
    parsed = urllib.parse.urlparse(expanded)
    query_params = urllib.parse.parse_qs(parsed.query)

    if "query_place_id" in query_params:
        return {"place_id": query_params["query_place_id"][0]}

    if "q" in query_params:
        q = query_params["q"][0]
        if q.startswith("place_id:"):
            return {"place_id": q.replace("place_id:", "")}
        else:
            return {"text_query": urllib.parse.unquote(q)}

    if "ftid" in query_params and query_params["ftid"]:
        return {"text_query": query_params["ftid"][0]}

    if "destination" in query_params and query_params["destination"]:
        return {"text_query": urllib.parse.unquote_plus(query_params["destination"][0])}

    # path 裡面 /maps/place/... 也可以切出來當 text_query
    if "/maps/place/" in parsed.path:
        parts = parsed.path.split("/maps/place/")[-1].split("/")
        if parts:
            name_part = urllib.parse.unquote_plus(parts[0])
            return {"text_query": name_part}

    return {}
