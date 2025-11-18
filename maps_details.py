# ✅ 使用 Selenium 解析前端導向（JS/Meta Refresh/Firebase Dynamic Links）
#   - 僅在 requests 無法取得最終 URL 時才呼叫
#   - 以 lazy import 方式避免在未安裝 selenium 環境直接報錯

import sys
import json
import requests

from maps_common import (
    expand_maps_share_url,
    extract_place_info,
)


# ✅ 簡易偵錯輸出（只在需要時呼叫）


def _debug_print_api(name: str, payload: dict):
    try:
        status = payload.get("status")
        err = payload.get("error_message")
        candidates = payload.get("candidates")
        results = payload.get("results")
        print(f"[DEBUG] {name} status={status} candidates={len(candidates) if isinstance(candidates, list) else None} results={len(results) if isinstance(results, list) else None} error={err}")
    except Exception:
        pass

# ✅ 追蹤分享短鏈並展開至最終 URL


# ✅ 讀取設定檔


def load_api_key(config_path="config.json"):
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return config.get("GOOGLE_MAPS_API_KEY")
    except FileNotFoundError:
        print("❌ 找不到 config.json，請確認檔案是否存在")
        sys.exit(1)

# ✅ 取得 place_id


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

    response = requests.get(url, params=params, timeout=20).json()
    _debug_print_api("findplace", response)
    if response.get("candidates"):
        return response["candidates"][0]["place_id"]
    # retry without locationbias once
    if latlng:
        params.pop("locationbias", None)
        response = requests.get(url, params=params, timeout=20).json()
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

# ✅ Fallback: Text Search 近似查找


def text_search_place_id(api_key, query, latlng=None, radius=2000):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"key": api_key, "query": query}
    if latlng:
        lat, lng = latlng.split(",")
        params.update({"location": latlng, "radius": radius})
    resp = requests.get(url, params=params, timeout=20).json()
    _debug_print_api("textsearch", resp)
    if resp.get("results"):
        return resp["results"][0].get("place_id")
    return None

# ✅ Fallback: Nearby Search with keyword（搭配座標）


def nearby_search_place_id(api_key, keyword, latlng, radius=2000):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {"key": api_key, "keyword": keyword,
              "location": latlng, "radius": radius}
    resp = requests.get(url, params=params, timeout=20).json()
    _debug_print_api("nearbysearch", resp)
    if resp.get("results"):
        return resp["results"][0].get("place_id")
    return None

# ✅ 取得店家詳細資訊 + 評論


def get_place_details(api_key, place_id):
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": api_key,
        "fields": "all",
        "reviews_no_translations": "true",
        "reviews_sort": "newest",
    }
    return requests.get(url, params=params).json()

# ✅ 輸出結果


def display_details(result):
    if result.get("status") != "OK":
        print("❌ API Error:", result.get("status"), result.get("error_message"))
        return

    r = result["result"]
    print("\n✅ 店家資訊：")
    print("🏷 名稱：", r.get("name"))
    print("📍 地址：", r.get("formatted_address"))
    print("📞 電話：", r.get("formatted_phone_number"))
    print("⭐ 評分：", r.get("rating"), "/", r.get("user_ratings_total"))
    print("🌐 網站：", r.get("website"))
    print("🗺 Google 地圖：", r.get("url"))

    print("\n🕒 營業時間：")
    weekday_text = (r.get("opening_hours") or {}).get("weekday_text")
    if weekday_text:
        for line in weekday_text:
            print("  -", line)
    else:
        print("  (未提供)")

    print("\n📝 最新評論（最多 5 筆）：")
    for review in r.get("reviews", []):
        print("\n作者：", review.get("author_name"))
        print("評分：", review.get("rating"))
        print("時間：", review.get("relative_time_description"))
        print("內容：", review.get("text")[:200], "...")

# ✅ 主程式


def main():
    if len(sys.argv) < 2:
        print("用法：python script.py 'Google Maps URL'")
        sys.exit(1)

    api_key = load_api_key()

    place_info = extract_place_info(sys.argv[1])
    print("🔍 抽取資訊：", place_info)
    expanded_preview = expand_maps_share_url(sys.argv[1])
    print("🔎 展開後網址：", expanded_preview)
    if expanded_preview == sys.argv[1].strip():
        print("⚠️ 未能自動展開短連結；已啟用 HTML 解析 fallback。如仍失敗請將最終跳轉頁面貼上。")

    if not place_info.get("place_id") and not place_info.get("text_query"):
        # 嘗試用展開後的網址當作文字查詢
        place_info["text_query"] = expanded_preview

    place_id = place_info.get("place_id")

    if not place_id and "text_query" in place_info:
        place_id = find_place_id(
            api_key, place_info["text_query"], place_info.get("latlng"))

    # 若仍找不到，可能是 API Key 限制或計費/配額問題
    if not place_id and expanded_preview.startswith("https://www.google."):
        print("[提示] 若瀏覽器可開啟但 API 皆回 ZERO_RESULTS/REQUEST_DENIED：\n - 請確認使用的金鑰已啟用 Places API\n - 金鑰應為伺服器用（IP 限制），不可用只限 HTTP referrer 的前端金鑰\n - 將這個查詢改用 Text Search + location/radius 通常可解")

    if not place_id:
        print("❌ 無法取得 place_id。建議：")
        print("  1) 使用 /maps/place/ 開頭的店家頁連結（非純座標或路線分享）。")
        print("  2) 若是 maps.app.goo.gl 或 goo.gl/maps 短鏈，請提供展開後的最終連結。")
        print("  3) 或改提供：店名 + 地址（我會用 Find Place 解析）。")
        sys.exit(1)

    result = get_place_details(api_key, place_id)
    print(result)
    # display_details(result)


if __name__ == "__main__":
    main()
