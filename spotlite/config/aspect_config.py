# spotlite/co/aspect_config.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import logging

from spotlite.config.config import DOMAINS_ROOT
from spotlite.utils.io_utils import load_json

logger = logging.getLogger(__name__)


# 全域快取，避免重複 load
DOMAIN_CACHE: Dict[str, Dict[str, Any]] = {}


def _get_domain_paths(domain: str) -> Dict[str, Path]:
    """
    回傳該 domain 底下三個必要設定檔路徑：
      - stopwords.json
      - protected_phrases.json
      - aspect_seeds.json
    """
    base = DOMAINS_ROOT / domain
    return {
        "stopwords": base / "stopwords.json",
        "protected_phrases": base / "protected_phrases.json",
        "aspect_seeds": base / "aspect_seeds.json",
    }


def init_domain(domain: str) -> Dict[str, Any]:
    """
    初始化某個 domain 的設定：
      - 讀取 stopwords.json
      - 讀取 protected_phrases.json
      - 讀取 aspect_seeds.json（每個 aspect 底下有 seeds_pos / seeds_neg）
      - 合併正負 → aspect_seeds（用於面向判斷）
      - 拆出 seeds_pos / seeds_neg（用於情緒判斷）
    並存到 DOMAIN_CACHE。
    """
    if domain in DOMAIN_CACHE:
        # 已載入 → 直接回傳
        return DOMAIN_CACHE[domain]

    paths = _get_domain_paths(domain)
    for key, p in paths.items():
        if not p.exists():
            logger.error("Domain config missing: %s → %s", key, p)
            raise FileNotFoundError(f"Missing domain config file: {p}")

    # Load JSON files
    stopwords = load_json(paths["stopwords"]) or []
    protected_phrases = load_json(paths["protected_phrases"]) or {}
    raw_seeds = load_json(paths["aspect_seeds"]) or {}

    # 預期格式：
    # {
    #   "taste": { "seeds_pos": [...], "seeds_neg": [...] },
    #   "service": { "seeds_pos": [...], "seeds_neg": [...] },
    #   ...
    # }
    if not isinstance(raw_seeds, dict) or not raw_seeds:
        raise ValueError(
            f"aspect_seeds.json for domain '{domain}' 必須是一個非空的物件，"
            "格式類似 {\"taste\": {\"seeds_pos\": [...], \"seeds_neg\": [...]}, ...}"
        )

    seeds_pos: Dict[str, list] = {}
    seeds_neg: Dict[str, list] = {}

    for aspect, group in raw_seeds.items():
        if not isinstance(group, dict):
            logger.warning("Aspect '%s' seeds 格式異常（非物件），將忽略該面向", aspect)
            continue

        pos_list = group.get("seeds_pos", []) or []
        neg_list = group.get("seeds_neg", []) or []

        # 確保是 list
        if not isinstance(pos_list, list):
            logger.warning("Aspect '%s' seeds_pos 不是 list，已強制轉成空 list", aspect)
            pos_list = []
        if not isinstance(neg_list, list):
            logger.warning("Aspect '%s' seeds_neg 不是 list，已強制轉成空 list", aspect)
            neg_list = []

        seeds_pos[aspect] = pos_list
        seeds_neg[aspect] = neg_list

    if not seeds_pos and not seeds_neg:
        raise ValueError(
            f"aspect_seeds.json for domain '{domain}' 解析後沒有任何 seeds_pos / seeds_neg，"
            "請檢查設定格式。"
        )

    # 合併正負面 seeds：用於「判斷這個 phrase 屬於哪個 aspect」
    merged_aspect_seeds: Dict[str, list] = {}
    for aspect in set(list(seeds_pos.keys()) + list(seeds_neg.keys())):
        merged: list = list(
            set(seeds_pos.get(aspect, []) + seeds_neg.get(aspect, []))
        )
        merged_aspect_seeds[aspect] = merged

    # 可用的 aspect 名稱
    aspect_list = sorted(merged_aspect_seeds.keys())

    DOMAIN_CACHE[domain] = {
        "stopwords": stopwords,
        "protected_phrases": protected_phrases,
        "seeds_pos": seeds_pos,
        "seeds_neg": seeds_neg,
        "aspect_seeds": merged_aspect_seeds,
        "aspects": aspect_list,
        "paths": paths,
    }

    logger.info("Initialized domain=%s, aspects=%s", domain, aspect_list)
    return DOMAIN_CACHE[domain]


def get_domain_config(domain: str) -> Dict[str, Any]:
    """取得 domain 設定（若尚未 init，會自動 init）"""
    if domain not in DOMAIN_CACHE:
        return init_domain(domain)
    return DOMAIN_CACHE[domain]
