import re
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging

import pandas as pd
from bs4 import BeautifulSoup as Soup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import WebDriverException

# === Added imports for structured parsing ===
from typing import Any, Dict

from spotlite.utils.io_utils import save_json, save_csv
from spotlite.crawler.google_maps.browser_utils import (
    create_en_browser,
    ensure_hl_en,
    expand_maps_share_url,
)

from spotlite.config.config import load_config

CRAWLER_CFG = load_config("crawler.json")
GENERAL_CFG = load_config("configs.json")

REVIEWS_CFG = CRAWLER_CFG.get("providers", {}).get(
    "google_maps", {}).get("reviews", {})
SELENIUM_CFG = CRAWLER_CFG.get("selenium", {})
BROWSER_CFG = SELENIUM_CFG.get("browser", {})

PATH_CFG = GENERAL_CFG.get("paths", {})

logger = logging.getLogger(__name__)

# ==============================
# Constants / selectors
# ==============================

STAR_SEL = '[aria-label$="stars"], [aria-label$="star"], [aria-label*="stars"], [aria-label*="star"]'

CONTAINER_SELECTORS = [
    # Primary scrollable reviews panel
    'div.m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde',
    'div.m6QErb.Pf6ghf.XiKgde.KoSBEe.ecceSd.tLjsW',
    'div.m6QErb.XiKgde.tLjsW',
    'div.m6QErb[aria-label][jslog]',  # Fallback
]

REVIEW_NODE_SELECTOR = 'div[data-review-id][jsaction*="review.in"]'
DATE_SELECTOR = '.rsqaWe, .section-review-publish-date'
STRUCTURED_BLOCK_SELECTOR = 'div[jslog="127691"]'

# Stop scrolling once we see "2 years ago" or older (e.g., 3 years ago)
STOP_AT_YEARS_AGO = REVIEWS_CFG.get("stop_at_years_ago", 2)

# ==============================
# Browser helpers
# ==============================

OUTPUT_ROOT = Path(PATH_CFG.get(
    "google_map_reviews_output_root", "data/google_map/reviews"))
SAVE_JSON = REVIEWS_CFG.get("save_json", True)


def click_safely(browser, elem):
    """
    Click an element safely:
      - Scroll into view
      - Click via normal click / JS click fallback
      - If a new tab / window opens, close it and return focus.
    """
    base_handles = set(browser.window_handles)
    browser.execute_script(
        'arguments[0].scrollIntoView({block:"center"});', elem
    )
    try:
        elem.click()
    except WebDriverException:
        browser.execute_script('arguments[0].click();', elem)
    time.sleep(0.8)

    try:
        now_handles = set(browser.window_handles)
        extras = list(now_handles - base_handles)
        for h in extras:
            browser.switch_to.window(h)
            try:
                browser.close()
            except WebDriverException:
                pass
        if extras:
            browser.switch_to.window(list(base_handles)[0])
    except WebDriverException:
        pass


def open_reviews_tab(browser):
    """Ensure we are on the Reviews tab."""
    try:
        reviews_tab = WebDriverWait(browser, 12).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    'button[role="tab"][aria-label^="Reviews"], '
                    'button[aria-label*="Reviews"]',
                )
            )
        )
        click_safely(browser, reviews_tab)
        time.sleep(1.0)
    except WebDriverException as e:
        logger.warning(
            "⚠️ Reviews tab not found; maybe already on Reviews or selector needs update (%s)",
            e,
        )


def sort_reviews_newest(browser):
    """Click the Sort button and choose 'Newest' if available."""
    try:
        sort_btn = WebDriverWait(browser, 10).until(
            EC.element_to_be_clickable(
                (
                    By.CSS_SELECTOR,
                    'button[aria-label="Sort reviews"], '
                    'button[aria-label*="Sort reviews"]',
                )
            )
        )
        click_safely(browser, sort_btn)
        time.sleep(0.8)

        # Wait for the sort menu to appear
        menu_root = WebDriverWait(browser, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, 'div[role="menu"]'))
        )

        # In the menu, options are role="menuitemradio" with text inside .mLuXec
        options = menu_root.find_elements(
            By.CSS_SELECTOR, 'div[role="menuitemradio"]')
        target = None
        for opt in options:
            try:
                label_el = opt.find_element(By.CSS_SELECTOR, '.mLuXec')
            except Exception:
                label_el = opt
            txt = (label_el.text or "").strip().lower()
            if txt == "newest" or "newest" in txt:
                target = opt
                break

        if target is not None:
            browser.execute_script(
                'arguments[0].scrollIntoView({block:"center"});', target
            )
            target.click()
            time.sleep(1.0)
            logger.info("✅ Sorted reviews by newest.")
        else:
            logger.warning(
                "⚠️ Could not find 'Newest' option in sort menu; using default order.")
    except WebDriverException as e:
        logger.warning(
            "⚠️ Sort button or menu not found; proceeding without changing sort order. (%s)",
            e,
        )


def find_review_container(browser):
    """Try several known selectors to locate the scrollable review container."""
    for css in CONTAINER_SELECTORS:
        try:
            container = WebDriverWait(browser, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, css))
            )
            logger.info("✅ Review container found via: %s", css)
            return container
        except WebDriverException:
            continue

    logger.warning(
        "⚠️ Could not locate dedicated scroll container; will use window scrolling as fallback.")
    return None


# ==============================
# Date parsing / normalization
# ==============================


def _subtract_days_from_today(n: int) -> str:
    return (datetime.today() - timedelta(days=n)).strftime("%Y-%m-%d")


def normalize_review_date(raw_date: str) -> str:
    """
    Normalize Google Maps relative date strings into YYYY-MM-DD (approximate).
    Keeps original string if parsing fails.
    """
    if not raw_date:
        return raw_date

    raw = raw_date.strip()
    lowered = raw.lower()

    # Handle "Edited 2 days ago"
    if lowered.startswith("edited "):
        lowered = lowered[7:].strip()

    # Hours ago → treat as today (or within last 1–2 days)
    if "hour" in lowered:
        m = re.search(r"(\d+)", lowered)
        if m:
            hours = int(m.group(1))
            dt = datetime.today() - timedelta(hours=hours)
            return dt.strftime("%Y-%m-%d")

    # Days
    if "day" in lowered:
        m = re.search(r"(\d+)", lowered)
        if m:
            return _subtract_days_from_today(int(m.group(1)))

    # Weeks
    if "week" in lowered:
        m = re.search(r"(\d+)", lowered)
        if m:
            return _subtract_days_from_today(int(m.group(1)) * 7)

    # Months (approximate as the 1st of that month)
    if "month" in lowered:
        m = re.search(r"(\d+)", lowered)
        if m:
            months_ago = int(m.group(1))
            y = datetime.today().year
            mo = datetime.today().month - months_ago
            while mo <= 0:
                y -= 1
                mo += 12
            return f"{y}-{mo:02d}-01"

    # Years (approximate as Jan 1 of that year)
    if "year" in lowered:
        m = re.search(r"(\d+)", lowered)
        if m:
            years_ago = int(m.group(1))
            y = datetime.today().year - years_ago
            return f"{y}-01-01"

    # Try absolute date like "Feb 3, 2023"
    try:
        return datetime.strptime(raw, "%b %d, %Y").strftime("%Y-%m-%d")
    except Exception:
        return raw_date


# ==============================
# Review parsing helpers
# ==============================


def extract_structured_summary(soup: Soup) -> str:
    """
    Extract the structured review block like:
      Service / Meal type / Price per person / Food: 5 / Service: 5 / ...
    and format as multi-line text.
    """
    structured_div = soup.select_one(STRUCTURED_BLOCK_SELECTOR)
    if structured_div is None:
        return ""

    parts = []
    # Each direct child block corresponds to one field group
    for block in structured_div.find_all('div', recursive=False):
        texts = [t.strip() for t in block.stripped_strings if t.strip()]
        if not texts:
            continue

        # Case 1: label + value, e.g. ["Service", "Dine in"]
        if len(texts) >= 2 and ':' not in texts[0]:
            label = texts[0]
            value = texts[1]
            entry = f"{label}: {value}"
        # Case 2: already "Food: 5"
        elif len(texts) == 1 and ':' in texts[0]:
            entry = texts[0]
        else:
            entry = ' '.join(texts)

        parts.append(entry)

    return '\n'.join(parts) if parts else ""


def parse_review_element(el) -> dict:
    """Parse a single Selenium WebElement (review) into a Python dict."""
    rid = el.get_attribute('data-review-id') or ''
    html = el.get_attribute('outerHTML')
    s = Soup(html, 'lxml')

    reviewer_el = s.select_one('.d4r55.fontTitleMedium')
    star_el = s.select_one(STAR_SEL)
    date_el = s.select_one(DATE_SELECTOR)
    raw_date = date_el.get_text(strip=True) if date_el else ''
    normalized_date = normalize_review_date(raw_date)

    # Structured details (Service, Meal type, Price per person, etc.)
    structured_summary = extract_structured_summary(s)

    # Main review text block (plain text only, without structured block)
    text_root = s.select_one('.MyEned, .section-review-text')
    if text_root is not None:
        # Remove the structured block so "Food: 1 Service: 1 ..." does not leak into plain_text
        for sub in text_root.select(STRUCTURED_BLOCK_SELECTOR):
            sub.decompose()
        plain_text = text_root.get_text(" ", strip=True)
    else:
        plain_text = ""

    # Full text = plain text + structured summary
    if structured_summary:
        full_text = f"{plain_text}\n\n{structured_summary}" if plain_text else structured_summary
    else:
        full_text = plain_text

    # Parse structured block into fields
    structured_block = structured_summary or ""
    structured_fields = {}
    patterns = {
        "meal_type": r"Meal type:\s*(.+)",
        "price_per_person": r"Price per person:\s*\$?(.+)",
        "food_score": r"Food:\s*([0-9])",
        "service_score": r"Service:\s*([0-9])",
        "atmosphere_score": r"Atmosphere:\s*([0-9])",
        "noise_level": r"Noise level:\s*(.+)",
        "wait_time": r"Wait time:\s*(.+)",
        "parking_space": r"Parking space:\s*(.+)",
        "parking_options": r"Parking options:\s*(.+)",
        "recommended_dishes": r"Recommended dishes?:\s*(.+)",
        "vegetarian_options": r"Vegetarian options?:\s*(.+)",
        "dietary_restrictions": r"Dietary restrictions?:\s*(.+)",
        "parking_general": r"Parking:\s*(.+)",
        "kid_friendliness": r"Kid[- ]friendliness?:\s*(.+)",
        "wheelchair_accessibility": r"Wheelchair accessibility:\s*(.+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, structured_block, flags=re.IGNORECASE)
        if not m:
            continue
        val = m.group(1).strip()
        if key.endswith("_score"):
            try:
                structured_fields[key] = int(val)
            except ValueError:
                structured_fields[key] = None
        elif key == "recommended_dishes":
            dishes = [x.strip() for x in re.split(r"[，,;/]", val) if x.strip()]
            structured_fields[key] = dishes
        else:
            structured_fields[key] = val

    # Parse stars into integer if possible
    stars_int = None
    if star_el is not None and star_el.has_attr('aria-label'):
        star_label = star_el['aria-label'].strip()
        m = re.search(r"(\d+)", star_label)
        if m:
            try:
                stars_int = int(m.group(1))
            except ValueError:
                stars_int = None

    data: Dict[str, Any] = {
        "review_id": rid,
        "reviewer": reviewer_el.get_text(strip=True) if reviewer_el else "",
        "stars": stars_int,
        "date": normalized_date,
        "raw_text": full_text,
        "plain_text": plain_text,
    }

    # Map structured fields to flat keys
    data.update({
        "meal_type": structured_fields.get("meal_type"),
        "price_per_person": structured_fields.get("price_per_person"),
        "food": structured_fields.get("food_score"),
        "service": structured_fields.get("service_score"),
        "atmosphere": structured_fields.get("atmosphere_score"),
        "noise_level": structured_fields.get("noise_level"),
        "wait_time": structured_fields.get("wait_time"),
        "parking": structured_fields.get("parking_general"),
        "parking_space": structured_fields.get("parking_space"),
        "parking_options": structured_fields.get("parking_options"),
        "recommended_dishes": structured_fields.get("recommended_dishes"),
        "vegetarian_options": structured_fields.get("vegetarian_options"),
        "dietary_restrictions": structured_fields.get("dietary_restrictions"),
        "kid_friendliness": structured_fields.get("kid_friendliness"),
        "wheelchair_accessibility": structured_fields.get("wheelchair_accessibility"),
    })

    return data


# ==============================
# Scrolling and expansion logic
# ==============================


def expand_visible_more_buttons(browser):
    """
    Expand all visible 'More' buttons for reviews.
    This is called repeatedly during scrolling so that
    by the end, all long reviews are already expanded.
    """
    try:
        # Include generic review "More" buttons by jsaction / aria-label,
        # plus the common w8nwRe kyuRq class used on many review expanders.
        more_btns = browser.find_elements(
            By.CSS_SELECTOR,
            'button[jsaction*="review.expandReview"], '
            'button[aria-label*="See more"], '
            'button.w8nwRe.kyuRq'
        )
        for btn in more_btns:
            try:
                # Only consider buttons that are tied to a specific review
                review_id = btn.get_attribute("data-review-id")
                if not review_id:
                    # Skip things like generic "more" in other UI
                    continue

                jsaction = (btn.get_attribute("jsaction") or "").lower()
                aria_label = (btn.get_attribute("aria-label") or "").lower()
                text_content = (btn.text or "").strip().lower()

                # Ensure this really is a review "More" button:
                # we require (1) jsaction contains review.expandReview,
                # (2) aria-label contains "see more", and
                # (3) visible text is "more"/"see more".
                if not (
                    "expandreview" in jsaction
                    and "see more" in aria_label
                    and text_content in ("more", "see more")
                ):
                    continue

                expanded = (btn.get_attribute("aria-expanded") or "").lower()
                if expanded == "true":
                    continue

                browser.execute_script(
                    'arguments[0].scrollIntoView({block:"center"});', btn
                )
                try:
                    btn.click()
                except WebDriverException:
                    browser.execute_script("arguments[0].click();", btn)
                time.sleep(0.3)
            except WebDriverException:
                continue
    except WebDriverException as e:
        logger.warning("⚠️ Error while expanding 'More' buttons: %s", e)


def should_stop_due_to_old_reviews(browser, years_threshold=STOP_AT_YEARS_AGO) -> bool:
    """
    Inspect current review nodes and return True if any review's displayed
    date text is 'N years ago' with N >= years_threshold.
    """
    try:
        nodes = browser.find_elements(By.CSS_SELECTOR, REVIEW_NODE_SELECTOR)
    except WebDriverException:
        return False

    for n in nodes:
        try:
            date_el = n.find_element(By.CSS_SELECTOR, DATE_SELECTOR)
            txt = (date_el.text or "").strip().lower()
            if "years ago" in txt:
                m = re.search(r"(\d+)", txt)
                if m and int(m.group(1)) >= years_threshold:
                    logger.info(
                        "⏹ Found old review with date '%s'; stop scrolling.", txt)
                    return True
        except WebDriverException:
            continue
    return False


def scroll_reviews_to_bottom(browser, container, max_rounds=600, pause=0.9):
    """
    Keep scrolling until new reviews stop loading OR we encounter reviews
    older than the configured years threshold.
    If container is None, scroll the window instead.
    """
    scroll_cfg = REVIEWS_CFG.get("scroll", {})
    stall_limit = scroll_cfg.get("stall_limit", 5)
    last_height = -1
    last_count = 0
    stall = 0

    for i in range(max_rounds):
        # Scroll container or window
        if container is not None:
            browser.execute_script(
                'arguments[0].scrollTop = arguments[0].scrollHeight', container
            )
            time.sleep(pause)
            h = browser.execute_script(
                'return arguments[0].scrollHeight', container
            )
        else:
            browser.execute_script(
                'window.scrollTo(0, document.body.scrollHeight);'
            )
            time.sleep(pause)
            h = browser.execute_script('return document.body.scrollHeight')

        # Expand any visible "More" review buttons
        expand_visible_more_buttons(browser)

        # Count review nodes in DOM
        try:
            nodes = browser.find_elements(
                By.CSS_SELECTOR, REVIEW_NODE_SELECTOR)
            count = len(nodes)
        except WebDriverException as e:
            logger.warning("⚠️ Error while finding review nodes: %s", e)
            nodes = []
            count = last_count

        # Stop early if we see "2 years ago" (or more) in any review
        if should_stop_due_to_old_reviews(browser, years_threshold=STOP_AT_YEARS_AGO):
            logger.info(
                "🔽 Scroll round %s: height=%s, reviews_loaded=%s", i+1, h, count)
            break

        logger.info(
            "🔽 Scroll round %s: height=%s, reviews_loaded=%s", i+1, h, count)

        # Stall detection: no growth in height or count
        if h == last_height and count == last_count:
            stall += 1
            if stall >= stall_limit:
                logger.info(
                    "⏹ No further growth in scroll height or review count; stop scrolling.")
                break
        else:
            stall = 0
            last_height = h
            last_count = count


# ==============================
# Main scraping flow
# ==============================


def scrape_reviews_for_url(raw_url: str):
    """
    Core entry point:
        - Create browser
        - Normalize / expand URL
        - Force English UI
        - Open Reviews tab
        - Sort by Newest
        - Scroll & expand
        - Parse all visible reviews
        - Return (place_name, [reviews])
    """
    browser = create_en_browser()

    try:
        # Normalize / expand share URL
        url = expand_maps_share_url(raw_url)
        browser.get(url)

        # Ensure hl=en on the resolved URL
        time.sleep(1)
        ensure_hl_en(browser)

        # Wait until page is interactive
        WebDriverWait(browser, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "body"))
        )

        # Go to reviews tab and sort
        open_reviews_tab(browser)
        sort_reviews_newest(browser)

        # Locate scrollable review container (if any)
        review_container = find_review_container(browser)
        container = review_container if review_container is not None else None

        logger.info("⏬ Scrolling reviews to bottom (auto 'More' expansion)...")
        scroll_cfg = REVIEWS_CFG.get("scroll", {})
        scroll_reviews_to_bottom(
            browser,
            container,
            max_rounds=scroll_cfg.get("max_rounds", 120),
            pause=scroll_cfg.get("pause_seconds", 0.8)
        )

        # Final pass: expand any remaining "More" buttons after scrolling
        expand_visible_more_buttons(browser)

        # After scrolling and final expansion, collect ALL visible review nodes once
        nodes = browser.find_elements(By.CSS_SELECTOR, REVIEW_NODE_SELECTOR)
        logger.info("📥 Found %s review nodes after scrolling.", len(nodes))

        parsed = []
        seen_ids = set()
        for el in nodes:
            rid = el.get_attribute('data-review-id') or ''
            if rid in seen_ids:
                continue
            d = parse_review_element(el)
            parsed.append(d)
            seen_ids.add(rid)

        logger.info(
            "✅ Parsed %s unique reviews from the current page view.", len(parsed))

        # Extract place name for filenames
        try:
            place_el = WebDriverWait(browser, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'h1.DUwDvf'))
            )
            place_name = place_el.text.strip()
        except WebDriverException:
            title = browser.title or "place"
            place_name = title.replace(" - Google Maps", "").strip()

        return place_name, parsed

    finally:
        try:
            browser.quit()
        except WebDriverException:
            pass


def save_reviews(place_name: str, reviews: list[dict]):
    """Save reviews to JSON and CSV using a sanitized place name as prefix."""
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", place_name) or "place"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Calculate unique dishes
    unique_dishes = set()
    for review in reviews:
        rd = review.get("recommended_dishes")
        if rd and isinstance(rd, list):
            for dish in rd:
                unique_dishes.add(dish)

    # Sort for consistent output
    dishes_list = sorted(list(unique_dishes))

    # JSON
    if SAVE_JSON:
        json_filename = OUTPUT_ROOT / f"{safe_name}_reviews.json"
        domain = REVIEWS_CFG.get("domain", "restaurant")
        json_payload = {
            "source": "google_maps",
            "place_name": safe_name,
            "domain": domain,
            "dishes": dishes_list,
            "reviews": reviews,
        }
        save_json(json_filename, json_payload)
        logger.info("💾 Saved reviews to %s", json_filename)
    else:
        logger.info("🔕 JSON saving disabled in config.")
