import re
import time
import pandas as pd
from bs4 import BeautifulSoup as Soup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

# ============ WebDriver bootstrap (English UI; do NOT auto-close) ============
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
options.add_argument("--lang=en-US")
options.add_experimental_option("prefs", {"intl.accept_languages": "en,en_US"})
browser = webdriver.Chrome(options=options)

# Best-effort: force Accept-Language headers
try:
    browser.execute_cdp_cmd("Network.enable", {})
    browser.execute_cdp_cmd("Network.setExtraHTTPHeaders", {
        "headers": {"Accept-Language": "en-US,en;q=0.9"}
    })
except Exception:
    pass

# -------- Target URL (edit this to your place) --------
url = "https://maps.app.goo.gl/qXabhrRhiFsKfE3k8"
browser.get(url)

# Ensure hl=en on the resolved URL
try:
    time.sleep(1)
    cur = browser.current_url
    if "hl=" not in cur:
        sep = "&" if "?" in cur else "?"
        browser.get(cur + f"{sep}hl=en&gl=US")
except Exception:
    pass

# Wait until page is interactive
WebDriverWait(browser, 20).until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "body")))

# ============ Helpers ============


def click_safely(elem):
    """Click an element; if a new tab/window opens, close it and return focus."""
    base = set(browser.window_handles)
    browser.execute_script(
        'arguments[0].scrollIntoView({block:"center"});', elem)
    try:
        elem.click()
    except Exception:
        browser.execute_script('arguments[0].click();', elem)
    time.sleep(0.8)
    try:
        now = set(browser.window_handles)
        extras = list(now - base)
        for h in extras:
            browser.switch_to.window(h)
            try:
                browser.close()
            except WebDriverException:
                pass
        if extras:
            browser.switch_to.window(list(base)[0])
    except WebDriverException:
        pass


# Open the Reviews tab (must do this first)
try:
    reviews_tab = WebDriverWait(browser, 12).until(
        EC.element_to_be_clickable(
            (By.CSS_SELECTOR, 'button[role="tab"][aria-label^="Reviews"], button[aria-label*="Reviews"]'))
    )
    click_safely(reviews_tab)
    time.sleep(1)
except Exception:
    print("⚠️ Reviews tab not found; maybe already on Reviews or selector needs update")

# Find the scrollable review container (prefer the one you provided)
review_container = None
CONTAINER_SELECTORS = [
    'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde',  # your provided scrollable panel
    'div.m6QErb.Pf6ghf.XiKgde.KoSBEe.ecceSd.tLjsW',
    'div.m6QErb.XiKgde.tLjsW',
    'div.m6QErb[aria-label][jslog]',  # fallback
]
for css in CONTAINER_SELECTORS:
    try:
        review_container = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css))
        )
        print(f"✅ Review container found via: {css}")
        break
    except Exception:
        continue
if review_container is None:
    print("⚠️ Could not locate dedicated scroll container; will use window scrolling as fallback.")

# Button finder (be strict to avoid wrong clicks)


def find_more_button():
    # Strict: look for visible button whose inner span text startswith 'More reviews'
    try:
        spans = browser.find_elements(
            By.CSS_SELECTOR, 'button.M77dve span.wNNZR')
        for sp in spans:
            txt = (sp.text or '').strip().lower()
            if txt.startswith('more reviews'):
                btn = sp.find_element(By.XPATH, './ancestor::button[1]')
                if btn.is_displayed() and btn.is_enabled():
                    return btn
    except Exception:
        pass
    # Fallbacks
    sel = (
        'button[aria-label^="More reviews"], '
        'button[jsaction*="pane.wfvdle337"], '
        'button[jsaction*="pane.wfvdle"][aria-label*="More reviews"]'
    )
    try:
        for b in browser.find_elements(By.CSS_SELECTOR, sel):
            if b.tag_name.lower() == 'button' and b.is_displayed() and b.is_enabled():
                return b
    except Exception:
        pass
    return None

# Scroll to bottom of the container until More button appears OR scrolling stalls


def scroll_until_more_or_stall(container, max_rounds=60, pause=0.9):
    last_height = -1
    stall = 0
    for _ in range(max_rounds):
        if find_more_button():
            return True
        browser.execute_script(
            'arguments[0].scrollTop = arguments[0].scrollHeight', container)
        time.sleep(pause)
        h = browser.execute_script(
            'return arguments[0].scrollHeight', container)
        if h == last_height:
            stall += 1
            if stall >= 3:
                return False
        else:
            stall = 0
        last_height = h
    return False


# Parse a single review element into dict
STAR_SEL = '[aria-label$="stars"], [aria-label$="star"], [aria-label*="stars"], [aria-label*="star"]'


def parse_review(el) -> dict:
    rid = el.get_attribute('data-review-id') or ''
    html = el.get_attribute('outerHTML')
    s = Soup(html, 'lxml')
    reviewer = s.select_one('.d4r55.fontTitleMedium')
    subtitle = s.select_one('.RfnDt')
    star_el = s.select_one(STAR_SEL)
    date_el = s.select_one('.rsqaWe, .section-review-publish-date')
    text_el = s.select_one('.MyEned, .section-review-text')
    photos = []
    for btn in s.select('.Tya61d, .section-review-photo'):
        style = btn.get('style', '')
        m = re.search(r'background-image:\s*url\(([^\)]+)\)', style)
        if m:
            url_ = m.group(1).strip().strip('"').strip("'")
            photos.append(url_)
    return {
        'review_id': rid,
        'reviewer': reviewer.get_text(strip=True) if reviewer else '',
        'subtitle': subtitle.get_text(strip=True) if subtitle else '',
        'stars': (star_el['aria-label'].strip() if star_el and star_el.has_attr('aria-label') else ''),
        'date': date_el.get_text(strip=True) if date_el else '',
        'text': text_el.get_text(' ', strip=True) if text_el else '',
        'photo_urls': photos,
    }


# ============ Main paging loop (your logic) ============
parsed = []
seen_ids = set()
container = review_container if review_container is not None else browser
round_idx = 1

while True:
    found_more = False
    if container is review_container:
        found_more = scroll_until_more_or_stall(
            review_container, max_rounds=60, pause=0.9)
    else:
        # Window-scrolling fallback (no dedicated container)
        last_h = -1
        for _ in range(12):
            if find_more_button():
                found_more = True
                break
            browser.execute_script(
                'window.scrollBy(0, document.body.scrollHeight);')
            time.sleep(1)
            h = browser.execute_script('return document.body.scrollHeight')
            if h == last_h:
                break
            last_h = h

    # Parse all currently loaded reviews (only NEW ones this round)
    nodes = browser.find_elements(
        By.CSS_SELECTOR, 'div[data-review-id][jsaction*="review.in"]')
    new_round = []
    for el in nodes:
        rid = el.get_attribute('data-review-id') or ''
        if rid in seen_ids:
            continue
        d = parse_review(el)
        parsed.append(d)
        seen_ids.add(rid)
        new_round.append(d)

    print(new_round)
    exit()
    # Print newly parsed reviews BEFORE clicking More (debug convenience)
    if new_round:
        print(
            f"\n===== Round {round_idx} — Newly parsed {len(new_round)} review(s) BEFORE clicking 'More reviews' =====")
        for i, d in enumerate(new_round, start=1):
            rid_short = (d.get('review_id', '') or '')[:12]
            stars = d.get('stars', '')
            date_ = d.get('date', '')
            who = d.get('reviewer', '')
            text_snip = (d.get('text', '') or '')
            if len(text_snip) > 200:
                text_snip = text_snip[:200] + '…'
            print(
                f"[{i}] id={rid_short} | stars={stars} | date={date_} | reviewer={who}")
            print(text_snip)
            if d.get('photo_urls'):
                print(f"photos: {d['photo_urls'][:3]}")
        print("===============================================================\n")
    else:
        print(
            f"\n===== Round {round_idx} — No NEW reviews found on this page before clicking 'More reviews' =====\n")

    print(
        f"📥 Parsed this round (visible nodes={len(nodes)}), total unique collected so far: {len(seen_ids)}")

    # If no More button is available, we're done
    btn = find_more_button()
    if not btn:
        print('ℹ️ No more "More reviews" button visible; stop paging.')
        break

    # Click More reviews → proceed to next round
    click_safely(btn)
    time.sleep(1.5)
    round_idx += 1

# ============ Final summary ============
df = pd.DataFrame(parsed)
print(f"✅ Done. Collected {len(df)} unique reviews in total.")
if not df.empty:
    print(df.head(3))
# Optional: save
# df.to_csv('reviews.csv', index=False)
