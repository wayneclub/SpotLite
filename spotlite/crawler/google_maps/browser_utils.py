from typing import Optional
from selenium import webdriver
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import WebDriverException

# Load selenium browser config from crawler.json
from spotlite.config.config import load_config
CRAWLER_CFG = load_config("crawler.json")
SELENIUM_CFG = CRAWLER_CFG.get("selenium", {})
BROWSER_CFG = SELENIUM_CFG.get("browser", {})

# ========= 共用：建立英文介面的 Chrome =========


def create_en_browser(headless: Optional[bool] = None) -> webdriver.Chrome:
    """
    建立一個預設就是英文 UI 的 Chrome WebDriver。
    reviews / details 都可以共用。
    """
    if headless is None:
        headless = BROWSER_CFG.get("headless", False)
    window_size = BROWSER_CFG.get("window_size", "1280,900")
    user_agent = BROWSER_CFG.get("user_agent", "")
    accept_lang = BROWSER_CFG.get("accept_language", "en-US,en;q=0.9")

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--window-size={window_size}")
    options.add_argument("--lang=en-US")
    options.add_experimental_option(
        "prefs", {"intl.accept_languages": accept_lang}
    )
    if user_agent:
        options.add_argument(f"--user-agent={user_agent}")

    browser = webdriver.Chrome(options=options)

    # Best-effort: force Accept-Language headers
    try:
        browser.execute_cdp_cmd("Network.enable", {})
        browser.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
            "headers": {"Accept-Language": accept_lang}
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
    """Expand a Google Maps short link and return final expanded URL.
    If expansion fails, return original URL.
    """
    browser = create_en_browser(headless=True)
    try:
        browser.set_page_load_timeout(25)
        browser.get(url)
        WebDriverWait(browser, 15).until(
            lambda d: "google.com/maps" in d.current_url or "google.co" in d.current_url
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
    """Return expanded Google Maps URL (no HTML)."""
    try:
        if url.startswith(("https://maps.app.goo.gl", "http://maps.app.goo.gl",
                           "https://goo.gl/maps", "http://goo.gl/maps")):
            return resolve_with_selenium(url)
        return url
    except Exception:
        return url
