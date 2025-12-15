# spotlite/analysis/google_maps_analyzer.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from spotlite.config.config import load_config, CONFIG_DIR

from spotlite.analysis.base_analyzer import ReviewAnalyzerBase
from spotlite.analysis.sources.google_maps import GoogleMapsAdapter


class GoogleMapsAnalyzer(ReviewAnalyzerBase):
    def __init__(self, domain: str = "restaurant") -> None:
        super().__init__(domain=domain)
        self.adapter = GoogleMapsAdapter()

    def _normalize_to_unified_schema(self, raw_obj: Any) -> Dict[str, Any]:
        schema = self.adapter.to_unified_schema(raw_obj)
        # 補 domain（如果 adapter 裡沒填）
        schema["domain"] = self.domain
        return schema

    def _postprocess_results(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save aspect phrase results to CSV based on config.

        We output two files (if the corresponding data exists):
          - <place_name>_aspect_phrases_tfidf.csv
          - <place_name>_aspect_targets_aggregated.csv
        """
        cfg = load_config(CONFIG_DIR / "configs.json")
        output_root = cfg.get("aspect_phrases_output_root",
                              "outputs/aspect_phrases")
        Path(output_root).mkdir(parents=True, exist_ok=True)

        # Use place_name for filenames; fall back to "analysis"
        place_name = result.get("place_name") or "analysis"
        # Sanitize to avoid illegal path characters
        safe_place_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in str(place_name)
        )

        UNWANTED_COLS = ["source", "domain", "place_id",
                         "place_name", "metadata", "n_reviews"]

        wrote_any = False

        tfidf_obj = result.get("aspect_phrases_tfidf")
        agg_obj = result.get("aspect_targets_aggregated")

        nested = result.get("aspect_phrases")
        if nested and isinstance(nested, dict):
            if tfidf_obj is None:
                tfidf_obj = nested.get("aspect_phrases_tfidf")
            if tfidf_obj is None:
                tfidf_obj = nested.get("tfidf")
            if agg_obj is None:
                agg_obj = nested.get("aspect_targets_aggregated")
            if agg_obj is None:
                agg_obj = nested.get("aggregated_targets")

        def to_df(obj):
            if isinstance(obj, pd.DataFrame):
                return obj
            elif isinstance(obj, list):
                return pd.DataFrame(obj)
            else:
                return None

        tfidf_df = to_df(tfidf_obj)
        if tfidf_df is not None and not tfidf_df.empty:
            cols_to_drop = [c for c in UNWANTED_COLS if c in tfidf_df.columns]
            if cols_to_drop:
                tfidf_df = tfidf_df.drop(columns=cols_to_drop)
            tfidf_path = Path(output_root) / \
                f"{safe_place_name}_aspect_phrases_tfidf.csv"
            tfidf_df.to_csv(tfidf_path, index=False)
            wrote_any = True

        agg_df = to_df(agg_obj)
        if agg_df is not None and not agg_df.empty:
            cols_to_drop = [c for c in UNWANTED_COLS if c in agg_df.columns]
            if cols_to_drop:
                agg_df = agg_df.drop(columns=cols_to_drop)
            agg_path = Path(output_root) / \
                f"{safe_place_name}_aspect_targets_aggregated.csv"
            agg_df.to_csv(agg_path, index=False)
            wrote_any = True

        # If neither structured list exists, fall back to previous behavior
        if not wrote_any:
            # Best-effort: dump whole result as a single-row CSV for debugging
            fallback_df = pd.DataFrame([result])
            fallback_path = Path(output_root) / \
                f"{safe_place_name}_aspect_phrases_fallback.csv"
            fallback_df.to_csv(fallback_path, index=False)

        return result
