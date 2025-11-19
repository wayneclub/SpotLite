# Using Selenium to resolve frontend-based redirects (JS/Meta Refresh/Firebase Dynamic Links)
# - Only invoked when requests cannot obtain the final URL
# - Lazy import to avoid errors when Selenium is not installed
import logging
import os
import re
import json
import requests

from spotlite.config import get_config

# Global configuration: load from configs.json / configs.example.json
CONFIG = get_config()
GOOGLE_CFG = CONFIG.get("google_maps", {})
DEFAULT_TIMEOUT = GOOGLE_CFG.get("timeout_seconds", 20)

DETAILS_CFG = CONFIG.get("details", {})
DETAILS_OUTPUT_ROOT = DETAILS_CFG.get("output_root", "data/details")
DETAILS_SAVE_JSON = DETAILS_CFG.get("save_json", True)

logger = logging.getLogger(__name__)


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
    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
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
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
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
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"key": api_key, "keyword": keyword,
              "location": latlng, "radius": radius}
    resp = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT).json()
    _debug_print_api("nearbysearch", resp)
    if resp.get("results"):
        return resp["results"][0].get("place_id")
    return None

# Get detailed place information + reviews


def get_place_details(api_key, place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    fields = GOOGLE_CFG.get("details_fields", "all")
    params = {
        "place_id": place_id,
        "key": api_key,
        "fields": fields,
        "reviews_no_translations": "true",
        "reviews_sort": "newest",
    }
    return requests.get(url, params=params, timeout=DEFAULT_TIMEOUT).json()

# Output results


def display_details(result):
    if result.get("status") != "OK":
        logger.error(
            f"❌ API Error: {result.get('status')} {result.get('error_message')}")
        return

    r = result["result"]
    logger.info("✅ Business Information:")
    logger.info(f"🏷 Name: {r.get('name')}")
    logger.info(f"📍 Address: {r.get('formatted_address')}")
    logger.info(f"📞 Phone: {r.get('formatted_phone_number')}")
    logger.info(f"⭐ Rating: {r.get('rating')} / {r.get('user_ratings_total')}")
    logger.info(f"🌐 Website: {r.get('website')}")
    logger.info(f"🗺 Google Maps: {r.get('url')}")

    logger.info("🕒 Opening Hours:")
    weekday_text = (r.get("opening_hours") or {}).get("weekday_text")
    if weekday_text:
        for line in weekday_text:
            logger.info(f"  - {line}")
    else:
        logger.info("  (Not provided)")

    logger.info("📝 Latest Reviews (up to 5):")
    for review in r.get("reviews", []):
        logger.info(f"Author: {review.get('author_name')}")
        logger.info(f"Rating: {review.get('rating')}")
        logger.info(f"Time: {review.get('relative_time_description')}")
        logger.info(f"Content: {review.get('text')[:200]} ...")


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
