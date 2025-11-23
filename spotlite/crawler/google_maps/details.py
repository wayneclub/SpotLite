# Using Selenium to resolve frontend-based redirects (JS/Meta Refresh/Firebase Dynamic Links)
# - Only invoked when requests cannot obtain the final URL
# - Lazy import to avoid errors when Selenium is not installed
import logging
import os
import re
import json
import urllib
import requests

from spotlite.config.config import load_config
from spotlite.crawler.google_maps.browser_utils import expand_maps_share_url

API_CFG = load_config("api.json")
CRAWLER_CFG = load_config("crawler.json")
GENERAL_CFG = load_config("configs.json")

GM_API = API_CFG["google_maps"]["endpoints"]
DETAILS_CFG = GM_API["details"]

DEFAULT_TIMEOUT = DETAILS_CFG.get("timeout_seconds", 20)
DETAILS_FIELDS = DETAILS_CFG.get("fields", "all")
DETAILS_URL = DETAILS_CFG.get("url")

PATH_CFG = GENERAL_CFG.get("paths", {})
DETAILS_OUTPUT_ROOT = PATH_CFG.get(
    "google_map_details_output_root", "data/google_map/details")

CRAWLER_DETAILS_CFG = CRAWLER_CFG.get("providers", {}).get(
    "google_maps", {}).get("details", {})
DETAILS_SAVE_JSON = CRAWLER_DETAILS_CFG.get("save_json", True)

logger = logging.getLogger(__name__)


def get_place_id(api_key: str, url: str):
    """Get place_id from Google Maps URL.

    Strategy:
    1. If query_place_id is in URL query, return it directly.
    2. Otherwise, extract place name (text_query) and latlng from URL, then call find_place_id().
    """
    expanded = expand_maps_share_url(url)
    parsed = urllib.parse.urlparse(expanded)
    query_params = urllib.parse.parse_qs(parsed.query)

    # 1) URL 本身就帶有 place_id 類資訊
    if "query_place_id" in query_params:
        return query_params["query_place_id"][0]

    # 有些 URL 可能用 q=place_id:XXX 或 q=店名
    text_query = None
    if "q" in query_params:
        raw_q = query_params["q"][0]
        if raw_q.startswith("place_id:"):
            return raw_q.replace("place_id:", "")
        else:
            text_query = urllib.parse.unquote_plus(raw_q)

    # 2) 從 /maps/place/XXX 路徑中抓店名作為 text_query
    if text_query is None and "/maps/place/" in parsed.path:
        name_part = parsed.path.split("/maps/place/")[-1].split("/")[0]
        text_query = urllib.parse.unquote_plus(name_part)

    # 3) 從 path 中取得經緯度 @lat,lng,...
    latlng = None
    m = re.search(r"@(-?\d+\.\d+),(-?\d+\.\d+)", parsed.path)
    if m:
        latlng = f"{m.group(1)},{m.group(2)}"

    # 若完全拿不到 text_query，只能放棄
    if not text_query:
        logger.error(f"❌ 無法從 URL 解析出可用的店名/text_query: {expanded}")
        return None

    # 4) 使用店名 + (可選) 經緯度呼叫 find_place_id
    place_id = find_place_id(api_key, text_query, latlng=latlng)
    if place_id:
        return place_id

    logger.error(
        f"❌ 使用 Find Place/Text Search 仍無法找到 place_id，text_query={text_query}, latlng={latlng}")
    return None


def _debug_print_api(name: str, payload: dict):
    try:
        status = payload.get("status")
        err = payload.get("error_message")
        candidates = payload.get("candidates")
        results = payload.get("results")
        logger.debug(f"{name} status={status} candidates={len(candidates) if isinstance(candidates, list) else None} results={len(results) if isinstance(results, list) else None} error={err}")
    except Exception:
        pass

# Resolve and expand shared short URLs to their final destination


# Retrieve place_id


def find_place_id(api_key, text_query, latlng=None):
    url = GM_API["find_place"]["url"]
    params = {
        "key": api_key,
        "input": text_query,
        "inputtype": "textquery",
        "fields": "place_id,name,formatted_address"
    }
    if latlng:
        params["locationbias"] = f"circle:2000@{latlng}"

    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT).json()
    _debug_print_api("findplace", response)
    if response.get("candidates"):
        return response["candidates"][0]["place_id"]
    # retry without locationbias once
    if latlng:
        params.pop("locationbias", None)
        response = requests.get(
            url, params=params, timeout=DEFAULT_TIMEOUT).json()
        _debug_print_api("findplace(nobias)", response)
        if response.get("candidates"):
            return response["candidates"][0]["place_id"]
    # Fallbacks
    pid = text_search_place_id(api_key, text_query, latlng=latlng)
    if pid:
        return pid
    if latlng:
        pid = nearby_search_place_id(api_key, text_query, latlng=latlng)
        if pid:
            return pid
    return None

# Fallback: Text Search approximate lookup


def text_search_place_id(api_key, query, latlng=None, radius=2000):
    url = GM_API["text_search"]["url"]
    params = {"key": api_key, "query": query}
    if latlng:
        lat, lng = latlng.split(",")
        params.update({"location": latlng, "radius": radius})
    resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT).json()
    _debug_print_api("textsearch", resp)
    if resp.get("results"):
        return resp["results"][0].get("place_id")
    return None

# Fallback: Nearby Search with keyword (using coordinates)


def nearby_search_place_id(api_key, keyword, latlng, radius=2000):
    url = GM_API["nearby_search"]["url"]
    params = {"key": api_key, "keyword": keyword,
              "location": latlng, "radius": radius}
    resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT).json()
    _debug_print_api("nearbysearch", resp)
    if resp.get("results"):
        return resp["results"][0].get("place_id")
    return None

# Get detailed place information + reviews


def get_place_details(api_key, place_id):
    url = DETAILS_URL
    fields = DETAILS_FIELDS
    params = {
        "place_id": place_id,
        "key": api_key,
        "fields": fields,
        "reviews_no_translations": "true",
        "reviews_sort": "newest",
    }
    return requests.get(url, params=params, timeout=DEFAULT_TIMEOUT).json()


def save_details(details, place_id):
    if not DETAILS_SAVE_JSON:
        logger.info("🔕 JSON saving disabled in config.")
        return
    if details.get("status") != "OK":
        logger.error(
            f"❌ API Error: {details.get('status')} {details.get('error_message')}")
        return
    os.makedirs(DETAILS_OUTPUT_ROOT, exist_ok=True)

    place_name = details.get("result", {}).get("name") or place_id or "place"
    safe_name = re.sub(r'[\\/*?:"<>|]', "_", place_name) or "place"
    filename = os.path.join(DETAILS_OUTPUT_ROOT, f"{safe_name}_details.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    logger.info(f"✅ Saved details to {filename}")
