# spotlite/utils/io_utils.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


# ------------------------------------------------------------
# 基本 JSON I/O
# ------------------------------------------------------------
def load_json(path: Path | str) -> Any:
    """Load a JSON file and return Python object."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path | str, obj: Any, indent: int = 2) -> None:
    """Save Python object as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


# ------------------------------------------------------------
# CSV I/O
# ------------------------------------------------------------
def save_csv(path: Path | str, rows: List[Dict[str, Any]]) -> None:
    """
    Save list of dict rows as CSV.
    Keys of the first row are used as header.
    """
    import csv

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("save_csv() received empty rows list.")

    headers = list(rows[0].keys())

    with p.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# 專門讀 "reviews.json" 格式的方法
# ------------------------------------------------------------
def load_reviews_json(path: Path | str) -> List[Dict[str, Any] | str]:
    """
    專門讀「包含 reviews」的 JSON。
    接受兩種格式：
    1. {"reviews": [...]}   # 統一 schema
    2. [...]                # 直接是 reviews list
    回傳 reviews list（不做清洗）
    """
    obj = load_json(path)

    if isinstance(obj, dict) and "reviews" in obj:
        return obj["reviews"]

    if isinstance(obj, list):
        return obj

    raise ValueError(f"Invalid reviews JSON structure: {path}")


# ------------------------------------------------------------
# Optional：儲存分析結果 / 保證資料夾存在
# ------------------------------------------------------------
def ensure_dir(path: Path | str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_analysis_output(result: Dict[str, Any], output_path: Path | str) -> None:
    """將 run_analysis() 的結果存成 JSON。"""
    save_json(output_path, result)
