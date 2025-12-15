# spotlite/analysis/base_analyzer.py
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, List

from spotlite.config.aspect_config import init_domain
from spotlite.analysis.aspect_phrases import AspectPhraseExtractor
# from spotlite.analysis.ai_summary import AISummaryGenerator


class ReviewAnalyzerBase(ABC):
    """
    Template method:
      run_analysis(url_or_path) -> 統一 pipeline
    子類只需要客製化：
      - _load_raw_from_source (如果你是吃 URL)
      - _normalize_to_unified_schema (不同 source -> 統一 JSON)
      - _postprocess_results (要不要做額外輸出)
    """

    def __init__(self, domain: str = "restaurant") -> None:
        self.domain = domain
        # 初始化該 domain 的 stopwords / aspect seeds / protected phrases
        init_domain(domain=self.domain)
        # 共用的 components
        self.aspect_extractor = AspectPhraseExtractor(domain=self.domain)
        # self.summary_generator = AISummaryGenerator(domain=self.domain)

    # ---------- Template Method ----------

    def run_analysis(self, source_input: dict) -> Dict[str, Any]:
        """
        source_input:
          - 原始 JSON dict（已經 load 好，未 normalize）
        回傳一個「最終結果 dict」：可直接給前端/API 用。
        """

        # 1) 取得「標準化前」raw data（直接傳入 JSON dict）
        raw_obj = source_input

        # 2) 轉成你的「統一 reviews schema」
        unified = self._normalize_to_unified_schema(raw_obj)

        # 3) 預處理 reviews（純文字 / 結構化拆開）
        plain_reviews = unified.get("reviews", [])
        # 4) aspect keywords 分析（plain reviews）
        aspect_outputs = self.aspect_extractor.run(plain_reviews)

        # 5) 統一整理成最後輸出的 dict
        result = {
            "source": unified.get("source") if isinstance(unified, dict) else None,
            "domain": self.domain,
            "place_id": unified.get("place_id") if isinstance(unified, dict) else None,
            "place_name": unified.get("place_name") if isinstance(unified, dict) else None,
            "metadata": unified.get("metadata", {}) if isinstance(unified, dict) else {},
            "n_reviews": len(plain_reviews),
            "aspect_phrases": aspect_outputs,
        }

        # 6) 給子類一個 chance 做額外處理/儲存
        result = self._postprocess_results(result)
        return result

    @abstractmethod
    def _normalize_to_unified_schema(self, raw_obj: Any) -> Dict[str, Any]:
        """
        不同 source → 統一 schema，該 schema 就是你現在用的 Holbox_reviews.json 風格
        """
        raise NotImplementedError

    def _postprocess_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        預設什麼都不做。有需要可以在 subclass 裡 override，例如：
          - 自動存成 JSON 檔到 outputs/
          - 加上一些 debug / trace 資訊
        """
        return result
