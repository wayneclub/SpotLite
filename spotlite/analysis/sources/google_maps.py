# spotlite/analysis/sources/google_maps.py
from __future__ import annotations
from typing import Any, Dict, List

from pathlib import Path
from spotlite.analysis.sources.base_source import BaseSourceAdapter


class GoogleMapsAdapter(BaseSourceAdapter):

    def to_unified_schema(
        self,
        raw_obj: Dict[str, Any] | List[Dict[str, Any]],
        source_path: str | None = None,
    ) -> Dict[str, Any]:
        """
        將「你自己爬下來的 Google Maps Review JSON」統一成後續 NLP 格式。

        支援兩種輸入格式：

        1) 舊版 / 原始爬蟲輸出：list of dicts
           [
             {
               "review_id": "...",
               "reviewer": "...",
               "stars": "5 stars" or 5,
               "date": "2025-10-27",
               "text": "..."
             },
             ...
           ]

        2) 新版 preprocess 後的 reviews.json：dict 包 reviews list
           {
             "source": "google_maps",
             "place_name": "Cafe Dulce (USC Village)",
             "domain": "restaurant",
             "reviews": [
               {
                 "review_id": "...",
                 "reviewer": "...",
                 "stars": 3,
                 "date": "2025-11-22",
                 "raw_text": "...原始評論...",
                 "plain_text": "...純文字評論...",
                 "...各種結構化欄位...": ...
               },
               ...
             ]
           }

        Output 統一格式：
        {
          "source": "google_maps",
          "domain": "<若有從輸入得到，否則 None>",
          "place_name": "<若有從輸入或檔名推斷>",
          "reviews": [ "pure text", "pure text", ... ]
        }
        """
        reviews_texts: List[str] = []
        place_name: str | None = None
        domain: str | None = None

        # Case 1: 新版 dict 格式，含 "reviews" 欄位
        if isinstance(raw_obj, dict) and isinstance(raw_obj.get("reviews"), list):
            place_name = raw_obj.get("place_name")
            domain = raw_obj.get("domain")

            for item in raw_obj["reviews"]:
                if not isinstance(item, dict):
                    continue
                # 優先使用 plain_text，其次 text，再次 raw_text
                txt = (
                    item.get("plain_text")
                    or item.get("text")
                    or item.get("raw_text")
                    or ""
                )
                if txt and txt.strip():
                    reviews_texts.append(txt)

        # Case 2: 舊版 list-of-dicts 格式
        elif isinstance(raw_obj, list):
            for item in raw_obj:
                if not isinstance(item, dict):
                    continue
                txt = (
                    item.get("plain_text")
                    or item.get("text")
                    or item.get("raw_text")
                    or ""
                )
                if txt and txt.strip():
                    reviews_texts.append(txt)

        else:
            # 其他格式一律視為錯誤
            raise ValueError(
                "GoogleMapsAdapter expects either a dict with 'reviews' key "
                "or a list of review dicts."
            )

        # 若 place_name 尚未有值，嘗試從檔名推斷
        if place_name is None and source_path:
            place_name = Path(source_path).stem.replace("_reviews", "")

        unified = {
            "source": "google_maps",
            "domain": domain,
            "place_name": place_name,
            "reviews": reviews_texts,
        }
        return unified
