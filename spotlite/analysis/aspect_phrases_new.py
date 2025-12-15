# spotlite/analysis/aspect_phrases.py
"""
Aspect phrase extraction + TF-IDF + aggregation

Pipeline:
Step 0. Review → 清洗 + Protected Phrases
Step 1. Tokenize + 停用字過濾（交給 clean_en + base/domain stopwords）
Step 2. 分五大面向（embedding + aspect seeds）
Step 3. 分正向 / 負向（用星等）
Step 4. 在每個 (aspect, sentiment) 子集合上，對 phrase 跑 TF-IDF
Step 5. 依 target_noun 聚合，產出可用於 AI summary 的結構化結果
"""

from __future__ import annotations
from typing import List, Set, TYPE_CHECKING
from typing import List, Set

import json
import re
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer, util
import spacy  # 用來做英文 POS tagging

from spotlite.config.config import load_config
from spotlite.analysis.utils_text import clean_en

logger = logging.getLogger(__name__)

# --------------------------------------------------
# Config loading
# --------------------------------------------------

# 專案 root 目錄（假設結構為 spotlite/xxx）
PROJECT_ROOT = Path(__file__).resolve().parents[1]

GENERAL_CFG = load_config("configs.json")      # 全域 config：paths, keywords 等
KEYWORDS_CFG = load_config("keywords.json")

PATH_CFG = GENERAL_CFG.get("paths", {})
# aspect 分析輸出根目錄，可以在 configs.json 裡自訂，否則預設到 project_root/outputs/aspect_phrases
OUTPUT_ROOT = PROJECT_ROOT / \
    PATH_CFG.get("aspect_output_root", "outputs/aspect_phrases")

# 關鍵字 / stopwords / aspect seeds 的多領域設定

DEFAULT_DOMAIN = KEYWORDS_CFG.get("default_domain", "restaurant")
DOMAINS_ROOT = KEYWORDS_CFG.get("domains_root", "spotlite/config/domains")


def _get_domain_paths(domain: str) -> Dict[str, Path]:
    """
    根據 domain 組合出該領域 stopwords / protected_phrases / aspect_seeds 的路徑。
    預設目錄結構：
      spotlite/config/domains/{domain}/stopwords.json
      spotlite/config/domains/{domain}/protected_phrases.json
      spotlite/config/domains/{domain}/aspect_seeds.json
    """
    base = PROJECT_ROOT.parent / DOMAINS_ROOT / domain
    return {
        "stopwords": base / "stopwords.json",
        "protected_phrases": base / "protected_phrases.json",
        "aspect_seeds": base / "aspect_seeds.json",
    }


# 這幾個會在 init_domain(...) 時依據不同領域被更新
STOPWORDS: set[str] = set(ENGLISH_STOP_WORDS)
PROTECTED_PHRASES: Dict[str, str] = {}


# RAW：保留原始 aspect_seeds.json 結構（含 seeds_pos / seeds_neg），給 phrase sentiment 用
ASPECT_SEEDS_RAW: Dict[str, Dict[str, List[str]]] = {}

# 合併版：pos+neg 合併後的種子，用來建立每個 aspect 的語意中心（assign_aspect 用）
ASPECT_SEEDS: Dict[str, List[str]] = {}


# 全域 opinion hints：補強 phrase_sentiment，避免過度依賴星等
# Global opinion seeds will be loaded from domain-specific aspect_seeds.json in init_domain()
GLOBAL_POS_SEEDS: List[str] = []
# Global opinion seeds will be loaded from domain-specific aspect_seeds.json in init_domain()
GLOBAL_NEG_SEEDS: List[str] = []


def init_domain(domain: Optional[str] = None) -> None:
    """
    載入指定領域的 stopwords / protected_phrases / aspect_seeds，
    並更新全域變數與 aspect centers。支援兩種 aspect_seeds.json 格式：

    1) 舊格式：
       {
         "aspects": {
           "taste": ["...", "..."],
           "service": [...]
         }
       }

    2) 新格式（推薦，含正負）：
       {
         "taste": {
           "seeds_pos": ["delicious", "tasty", ...],
           "seeds_neg": ["bland", "overcooked", ...]
         },
         "service": { ... }
       }
    """
    from typing import cast

    global STOPWORDS, PROTECTED_PHRASES, ASPECT_SEEDS, ASPECT_SEEDS_RAW, _ASPECT_CENTERS, GLOBAL_POS_SEEDS, GLOBAL_NEG_SEEDS

    if domain is None:
        domain = DEFAULT_DOMAIN

    paths = _get_domain_paths(domain)
    logger.info(
        f"Initializing aspect configs for domain='{domain}' from {paths}")

    # ---- Stopwords ----
    domain_stop: set[str] = set()
    stop_path = paths["stopwords"]
    if stop_path.exists():
        try:
            with open(stop_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                domain_stop.update(cast(List[str], data.get("stopwords", [])))
        except Exception as e:
            logger.error(f"Failed to load stopwords from {stop_path}: {e}")
    else:
        logger.warning(f"Domain stopwords file not found: {stop_path}")

    STOPWORDS = set(ENGLISH_STOP_WORDS).union(domain_stop)

    # ---- Protected phrases ----
    PROTECTED_PHRASES.clear()
    pp_path = paths["protected_phrases"]
    if pp_path.exists():
        try:
            with open(pp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                PROTECTED_PHRASES.update(data.get("protected_phrases", {}))
        except Exception as e:
            logger.error(
                f"Failed to load protected phrases from {pp_path}: {e}")
    else:
        logger.warning(f"Protected phrases file not found: {pp_path}")

    # ---- Aspect seeds ----
    ASPECT_SEEDS_RAW = {}
    ASPECT_SEEDS = {}

    asp_path = paths["aspect_seeds"]
    if asp_path.exists():
        try:
            with open(asp_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 格式 1：有 "aspects" key 的舊版
            if "aspects" in data:
                aspects = data.get("aspects", {})
                for aspect, seeds in aspects.items():
                    if not isinstance(seeds, list):
                        continue
                    seeds_list = [str(s) for s in seeds]
                    ASPECT_SEEDS[aspect] = seeds_list
                    ASPECT_SEEDS_RAW[aspect] = {
                        "seeds_pos": seeds_list,
                        "seeds_neg": [],
                    }
            else:
                # 格式 2：每個 aspect 底下有 seeds_pos / seeds_neg
                tmp_raw: Dict[str, Dict[str, List[str]]] = {}
                tmp_merged: Dict[str, List[str]] = {}
                for aspect, seeds_obj in data.items():
                    if not isinstance(seeds_obj, dict):
                        continue
                    pos = seeds_obj.get("seeds_pos", []) or []
                    neg = seeds_obj.get("seeds_neg", []) or []
                    pos = [str(s) for s in pos]
                    neg = [str(s) for s in neg]
                    tmp_raw[aspect] = {
                        "seeds_pos": pos,
                        "seeds_neg": neg,
                    }
                    tmp_merged[aspect] = pos + neg

                ASPECT_SEEDS_RAW = tmp_raw
                ASPECT_SEEDS = tmp_merged

            logger.info(
                f"Loaded aspect seeds for domain='{domain}': {list(ASPECT_SEEDS.keys())}")

            # Build GLOBAL opinion seeds from aspect_seeds.json (union across all aspects)
            pos_union: set[str] = set()
            neg_union: set[str] = set()
            for _asp, seeds_obj in ASPECT_SEEDS_RAW.items():
                for s in (seeds_obj.get("seeds_pos", []) or []):
                    pos_union.add(str(s))
                for s in (seeds_obj.get("seeds_neg", []) or []):
                    neg_union.add(str(s))

            GLOBAL_POS_SEEDS = sorted(pos_union)
            GLOBAL_NEG_SEEDS = sorted(neg_union)

            logger.info(
                "Loaded GLOBAL opinion seeds from aspect_seeds.json: pos=%d, neg=%d",
                len(GLOBAL_POS_SEEDS),
                len(GLOBAL_NEG_SEEDS),
            )

        except Exception as e:
            logger.error(f"Failed to load aspect seeds from {asp_path}: {e}")
    else:
        logger.warning(f"Aspect seeds file not found: {asp_path}")

    # 每次切換 domain 都需要重建 aspect centers
    _ASPECT_CENTERS = None

# --------------------------------------------------
# Step 0: Protected phrases + cleaning
# --------------------------------------------------


def apply_protected_phrases(text: str, mapping: Dict[str, str]) -> str:
    """
    將 multi-word expressions 例如 'kids meal' → 'kids_meal'
    這樣後續 clean_en / token 化時不會拆開。
    """
    s = text.lower()
    # 長字串優先替換，避免局部重疊
    phrases = sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True)
    for phrase, repl in phrases:
        pattern = r"\b" + re.escape(phrase) + r"\b"
        s = re.sub(pattern, repl, s)
    return s


def clean_review_text(raw: str) -> str:
    """
    專門給本 pipeline 用的清洗：
    1) 先做 protected_phrases
    2) 再做 clean_en（記得 clean_en 裡不要把 '_' 刪掉）
    """
    s = apply_protected_phrases(raw, PROTECTED_PHRASES)
    s = clean_en(s)
    return s

# --------------------------------------------------
# Step 2: Aspect assignment (embedding + seeds)
# --------------------------------------------------


_ASPECT_MODEL: Optional[SentenceTransformer] = None
_ASPECT_CENTERS = None


def get_aspect_model() -> SentenceTransformer:
    """
    get_aspect_model 的 Docstring

    :return: 說明
    :rtype: SentenceTransformer
    """
    global _ASPECT_MODEL
    if _ASPECT_MODEL is None:
        _ASPECT_MODEL = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2")
    return _ASPECT_MODEL


def build_aspect_centers():
    """
    將每個 aspect 的 seed 轉成 embedding centroid。
    """
    model = get_aspect_model()
    centers = {}
    for aspect, seeds in ASPECT_SEEDS.items():
        if not seeds:
            continue
        emb = model.encode(seeds, convert_to_tensor=True)
        centers[aspect] = emb.mean(dim=0)
    return centers


def assign_aspect(text: str, threshold: float = 0.35) -> Optional[str]:
    """
    對一個短文本（句子或 phrase）分配五大面向之一。
    回傳 None 表示沒有明顯屬於任何面向（可忽略）。
    """
    global _ASPECT_CENTERS
    if _ASPECT_CENTERS is None:
        _ASPECT_CENTERS = build_aspect_centers()

    if not _ASPECT_CENTERS:
        logger.warning("No aspect centers built; ASPECT_SEEDS may be empty.")
        return None

    model = get_aspect_model()
    v = model.encode(text, convert_to_tensor=True)

    best_aspect = None
    best_sim = -1.0
    for aspect, center in _ASPECT_CENTERS.items():
        sim = util.cos_sim(v, center).item()
        if sim > best_sim:
            best_sim = sim
            best_aspect = aspect

    if best_sim < threshold:
        return None
    return best_aspect

# --------------------------------------------------
# Step 3: Sentiment by rating
# --------------------------------------------------


def rating_to_sentiment(rating: float | int | None) -> Optional[str]:
    """
    用星等粗分 sentiment。
    4~5: pos, 1~2: neg, 3: 忽略。
    """
    if rating is None:
        return None
    try:
        r = float(rating)
    except Exception:
        return None
    if r >= 4:
        return "pos"
    if r <= 2:
        return "neg"
    return None  # 3 星就先不管

# --------------------------------------------------
# Phrase-level sentiment using seeds
# --------------------------------------------------


def phrase_sentiment(phrase: str, aspect: str, threshold: float = 0.1) -> Optional[str]:
    """
    對單一 phrase (e.g. 'too_sweet', 'not_good') 做情緒判斷。
    採用「規則優先 + AI 輔助」的混合策略。
    """
    # 0. 基礎設置
    if aspect not in ASPECT_SEEDS_RAW:
        return None

    ph_lower = phrase.lower()
    parts = ph_lower.split('_')

    # ------------------------------------------------------------------
    # 1. 規則判斷：處理 "too", "overly" (過度修飾)
    # ------------------------------------------------------------------
    excess_keywords = {"too", "overly", "excessively", "way"}  # way_too_...

    if any(k in parts for k in excess_keywords):
        # 定義「過量即壞事」的感官/物理屬性詞
        # 這些詞本身可能是中性或正面的(如 sweet)，但加上 too 就變負面
        SENSORY_NEGATIVE_IN_EXCESS = {
            # Taste
            "sweet", "salty", "sour", "bitter", "spicy", "dry", "hard", "tough",
            "cold", "hot", "greasy", "oily", "bland", "rich", "thick", "watery",
            # Price
            "expensive", "pricey", "costly", "high",
            # Environment / Service
            "loud", "noisy", "dark", "bright", "crowded", "small", "slow", "long", "fast"
        }

        # 定義「過量是好事/驚嘆」的評價詞
        EVALUATIVE_POSITIVE_IN_EXCESS = {
            "good", "great", "nice", "generous", "kind", "friendly", "helpful",
            "clean", "fresh", "cute", "beautiful", "fast"  # fast 視情況，但在服務通常是好事
        }

        # 檢查片語中的核心形容詞 (通常是最後一個字，例如 too_sweet 的 sweet)
        head_word = parts[-1]

        if head_word in SENSORY_NEGATIVE_IN_EXCESS:
            return "neg"  # too sweet -> neg
        elif head_word in EVALUATIVE_POSITIVE_IN_EXCESS:
            return "pos"  # too good -> pos

    # ------------------------------------------------------------------
    # 2. 規則判斷：處理 "not", "no", "never" (顯性否定)
    # ------------------------------------------------------------------
    negation_keywords = {"not", "no", "never", "didn",
                         "don", "wouldn", "couldn", "cant", "wont"}

    if any(k in parts for k in negation_keywords):
        # 簡單策略：如果有否定詞，先暫定為 neg
        # (雖然 not_bad 是好事，但在餐廳評論中，not good / not fresh 出現機率遠高於 not bad)
        # 進階做法可以是：如果核心詞是負面(bad)，則 not_bad -> neutral/pos

        head_word = parts[-1]
        NEGATIVE_ADJECTIVES = {"bad", "terrible", "awful", "horrible",
                               "gross", "disgusting", "rude", "slow", "dirty", "expensive"}

        if head_word in NEGATIVE_ADJECTIVES:
            return "pos"  # not bad, not expensive -> pos (或 neutral)
        else:
            return "neg"  # not good, not fresh -> neg

    # ------------------------------------------------------------------
    # 3. AI 語意相似度判斷 (Embedding Fallback)
    #    如果上面的規則都沒命中，才用 Embedding 算距離
    # ------------------------------------------------------------------
    seeds_obj = ASPECT_SEEDS_RAW.get(aspect, {})
    pos_seeds = list({str(s) for s in (seeds_obj.get(
        "seeds_pos", []) or [])} | set(GLOBAL_POS_SEEDS))
    neg_seeds = list({str(s) for s in (seeds_obj.get(
        "seeds_neg", []) or [])} | set(GLOBAL_NEG_SEEDS))

    if not pos_seeds and not neg_seeds:
        return None

    text = phrase.replace("_", " ")
    model = get_aspect_model()
    v = model.encode(text, convert_to_tensor=True)

    sims_pos = []
    sims_neg = []

    if pos_seeds:
        emb_pos = model.encode(pos_seeds, convert_to_tensor=True)
        sims_pos = util.cos_sim(v, emb_pos)[0].tolist()
    if neg_seeds:
        emb_neg = model.encode(neg_seeds, convert_to_tensor=True)
        sims_neg = util.cos_sim(v, emb_neg)[0].tolist()

    best_pos = max(sims_pos) if sims_pos else -1.0
    best_neg = max(sims_neg) if sims_neg else -1.0

    score = best_pos - best_neg

    if score > threshold:
        return "pos"
    elif score < -threshold:
        return "neg"

    return None

# def phrase_sentiment(phrase: str, aspect: str, threshold: float = 0.1) -> Optional[str]:
#     """
#     對單一 phrase (e.g. 'bland_soup') 做情緒判斷：
#       - 回傳 "pos" / "neg" / None（中立或不確定）

#     使用該 aspect 下的 seeds_pos / seeds_neg 做 cosine similarity 比較：
#       score = max_sim_pos - max_sim_neg
#       > threshold  → pos
#       < -threshold → neg
#       其餘         → None
#     """
#     if aspect not in ASPECT_SEEDS_RAW:
#         return None

#     seeds_obj = ASPECT_SEEDS_RAW.get(aspect, {})
#     pos_seeds = list({str(s) for s in (seeds_obj.get(
#         "seeds_pos", []) or [])} | set(GLOBAL_POS_SEEDS))
#     neg_seeds = list({str(s) for s in (seeds_obj.get(
#         "seeds_neg", []) or [])} | set(GLOBAL_NEG_SEEDS))

#     if not pos_seeds and not neg_seeds:
#         return None

#     text = phrase.replace("_", " ")  # "fresh_fish" -> "fresh fish"
#     model = get_aspect_model()
#     v = model.encode(text, convert_to_tensor=True)

#     sims_pos: List[float] = []
#     sims_neg: List[float] = []

#     if pos_seeds:
#         emb_pos = model.encode(pos_seeds, convert_to_tensor=True)
#         sims_pos = util.cos_sim(v, emb_pos)[0].tolist()

#     if neg_seeds:
#         emb_neg = model.encode(neg_seeds, convert_to_tensor=True)
#         sims_neg = util.cos_sim(v, emb_neg)[0].tolist()

#     best_pos = max(sims_pos) if sims_pos else -1.0
#     best_neg = max(sims_neg) if sims_neg else -1.0

#     score = best_pos - best_neg  # >0 越偏正向，<0 越偏負向

#     if score > threshold:
#         return "pos"
#     elif score < -threshold:
#         return "neg"
#     else:
#         return None

# --------------------------------------------------
# Phrase extraction (ADJ + NOUN)
# --------------------------------------------------


_NLP = None


def get_nlp():
    """
    準備 spaCy 英文模型：en_core_web_lg
    需先執行：
      python -m spacy download en_core_web_lg
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_lg")
    return _NLP


def extract_adj_noun_phrases(text: str) -> List[str]:
    """
    從已清洗的英文句子中抽出 ADJ+NOUN 片語，並轉成 'adj_noun' 形式。
    - Pattern 1: (ADV) + ADJ + NOUN (如 very_sweet_cake)
    - Pattern 2: NOUN + be + (ADV) + ADJ (如 very_sweet, 即使主語是 it)
    - Pattern 3: (ADV) + VERB + (ADV/ADJ) + (NOUN) (如 just_hit_different, not_recommend)
    - 強化過濾: Regex 清洗縮寫殘留 (don_t, not_sweeten_t)
    """
    # 確保取得 nlp 物件 (建議使用 en_core_web_lg 以獲得最佳效果)
    nlp = get_nlp()
    doc = nlp(text)

    phrases: List[str] = []

    # 定義有效詞性
    VALID_NOUN_POS = {"NOUN", "PROPN"}
    VALID_MODIFIER_POS = {"ADJ", "NUM", "PROPN"}

    # 1. 垃圾詞清單：這些詞無論在哪出現都應該視為雜訊 (解決分詞殘留)
    GARBAGE_STOPWORDS = {
        "don", "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
        "won", "wouldn", "shouldn", "couldn", "mustn", "mightn", "needn",
        "shan", "ain", "t", "s", "re", "ve", "m", "ll", "d", "em"
    }

    # 2. 否定與動詞停用清單：單獨出現時無意義
    NEGATION_STOPWORDS = {"not", "do", "be", "have", "will",
                          "can", "should", "get", "go", "say", "make", "allow", "taste"}

    # 合併檢查用的停用詞集
    CHECK_STOPWORDS = STOPWORDS.union(
        GARBAGE_STOPWORDS).union(NEGATION_STOPWORDS)

    # --- 輔助函數：用於獲取形容詞/動詞的副詞修飾語 ---
    def get_adv_modifier(token) -> str:
        """從一個 token 獲取其主要的 advmod 修飾語。"""
        adv_modifiers = []
        for child in token.children:
            if child.dep_ == "advmod" and child.pos_ == "ADV":
                lemma_lower = child.lemma_.lower()
                if lemma_lower not in CHECK_STOPWORDS:
                    adv_modifiers.append(lemma_lower)
        return adv_modifiers[0] if adv_modifiers else ""

    # --- Pattern 1: (ADV) + ADJ + NOUN (處理連續的形容詞 + 副詞修飾) ---
    for i in range(len(doc)):
        token = doc[i]

        # 尋找一個有效名詞
        if token.pos_ in VALID_NOUN_POS and token.lemma_.lower() not in CHECK_STOPWORDS:
            noun = token.lemma_.lower()
            adjective_chain = []

            # 向前迭代，尋找所有連在一起的有效修飾語
            j = i - 1
            while j >= 0 and doc[j].pos_ in VALID_MODIFIER_POS:
                current_token = doc[j]
                current_lemma = current_token.lemma_.lower()

                # 確保當前詞彙不是助動詞 (AUX) 或停用詞
                if current_token.pos_ != "AUX" and current_lemma not in CHECK_STOPWORDS:

                    # 1. 先插入形容詞
                    adjective_chain.insert(0, current_lemma)

                    # 2. 檢查這個形容詞是否有副詞修飾 (如 very -> sweet)
                    adv = get_adv_modifier(current_token)
                    if adv:
                        adjective_chain.insert(0, adv)

                    j -= 1
                else:
                    break

            # 如果找到形容詞或修飾語，則組合片語
            if adjective_chain:
                adj_part = "_".join(adjective_chain)
                phrases.append(f"{adj_part}_{noun}")

    # --- Pattern 2: NOUN + 'be' + ADV + ADJ (使用依存關係解析) ---
    for token in doc:
        if token.pos_ == "ADJ":
            adj = token.lemma_.lower()

            if token.dep_ in ("acomp", "attr"):
                copular_verb = token.head

                for child in copular_verb.children:
                    if child.dep_ == "nsubj" and child.pos_ in VALID_NOUN_POS:
                        noun = child.lemma_.lower()

                        is_negated = False
                        for verb_child in copular_verb.children:
                            if verb_child.dep_ == "neg" and verb_child.lemma_.lower() == "not":
                                is_negated = True
                                break

                        adv = get_adv_modifier(token)

                        if adj not in CHECK_STOPWORDS:
                            phrase_parts = []
                            if is_negated:
                                phrase_parts.append("not")
                            if adv:
                                phrase_parts.append(adv)
                            phrase_parts.append(adj)

                            # 處理代名詞主語 (如 "it was sweet")
                            if noun not in CHECK_STOPWORDS:
                                phrase_parts.append(noun)
                            else:
                                # 如果名詞是停用詞，但前面有修飾，保留形容詞部分
                                pass

                            # 確保片語夠長 (避免只提取出 'good')
                            if len(phrase_parts) > 1:
                                phrases.append("_".join(phrase_parts))
                            break

    # --- Pattern 3: (ADV) + VERB + (ADV/ADJ) + (NOUN) (捕捉複雜行為評價) ---
    # 改進：支援前置副詞 (just hit) 和 後置修飾 (hit different)
    for token in doc:
        if token.pos_ == "VERB":
            verb = token.lemma_.lower()

            pre_adv = ""    # 前置副詞 (如 just)
            post_attr = ""  # 後置補語/副詞 (如 different)
            is_negated = False
            noun_target = ""

            for child in token.children:
                # 1. 否定詞
                if child.dep_ == "neg" and child.lemma_.lower() == "not":
                    is_negated = True

                # 2. 前置副詞 (child 在 token 前面)
                elif child.dep_ == "advmod" and child.pos_ == "ADV" and child.i < token.i:
                    if child.lemma_.lower() not in CHECK_STOPWORDS:
                        pre_adv = child.lemma_.lower()

                # 3. 後置修飾 (child 在 token 後面，可能是副詞或形容詞補語)
                elif child.dep_ in ("advmod", "acomp") and child.i > token.i:
                    if child.lemma_.lower() not in CHECK_STOPWORDS:
                        post_attr = child.lemma_.lower()

                # 4. 名詞目標 (受詞或被動主語)
                elif child.dep_ in ("nsubjpass", "dobj") and child.pos_ in VALID_NOUN_POS:
                    if child.lemma_.lower() not in CHECK_STOPWORDS:
                        noun_target = child.lemma_.lower()

            # 核心過濾：動詞本身不能是停用詞
            if verb not in CHECK_STOPWORDS:
                phrase_parts = []

                if is_negated:
                    phrase_parts.append("not")
                if pre_adv:
                    phrase_parts.append(pre_adv)

                phrase_parts.append(verb)

                if post_attr:
                    phrase_parts.append(post_attr)
                if noun_target:
                    phrase_parts.append(noun_target)

                # 防止單詞被單獨提取 (長度必須 > 1)
                # 例如：防止提取出單獨的 "hit" 或 "make"
                if len(phrase_parts) > 1:
                    phrases.append("_".join(phrase_parts))

    # --- Regex 過濾器 (處理 tokenizer 產生的怪詞) ---
    # 1. 捕捉如 not_sweeten_t, don_t 這種以 _t 結尾的殘留
    contraction_artifact_1 = re.compile(r"^[a-z]+_t$")
    # 2. 捕捉如 we_re, i_m 這種代名詞縮寫
    contraction_artifact_2 = re.compile(
        r"^(?:i|we|you|they|it|that|there|what|who|let)_(?:m|re|s|ll|ve|d)$")

    # --- 最終強力過濾 ---
    final_phrases = []
    for phrase in phrases:
        ph_low = phrase.lower().strip()

        # Regex 過濾
        if contraction_artifact_1.fullmatch(ph_low) or contraction_artifact_2.fullmatch(ph_low):
            continue

        parts = ph_low.split('_')
        lemmas = [p for p in parts if p]

        # 單詞過濾：最後一道防線，確保沒有單獨的停用詞/垃圾詞
        if len(lemmas) == 1:
            single_word = lemmas[0]
            if single_word in CHECK_STOPWORDS:
                continue

        final_phrases.append(ph_low)

    return final_phrases

# 假設您在外部定義了 get_nlp() 和 STOPWORDS

# 為了程式碼的穩定性和可讀性，這裡使用 TYPE_CHECKING 註釋。
# 請確保在您的運行環境中：
# 1. get_nlp() 返回一個有效的 spaCy Language 實例 (最好是 en_core_web_lg)。
# 2. STOPWORDS 和 NEGATION_STOPWORDS 已經被正確定義。

# def extract_adj_noun_phrases(text: str) -> List[str]:
#     """
#     從已清洗的英文句子中抽出 ADJ+NOUN 片語，並轉成 'adj_noun' 形式。
#     - 包含 Pattern 1 (ADJ+NOUN) [升級：支援副詞修飾, 如 very_sweet], Pattern 2, Pattern 3
#     - 強化過濾單詞雜訊 (如 don't, isn't)
#     """
#     # 確保取得 nlp 物件 (建議使用 en_core_web_lg 以獲得最佳效果)
#     nlp = get_nlp()
#     doc = nlp(text)

#     phrases: List[str] = []

#     # 定義有效詞性
#     VALID_NOUN_POS = {"NOUN", "PROPN"}
#     VALID_MODIFIER_POS = {"ADJ", "NUM", "PROPN"}

#     # 核心過濾清單：單獨出現時無意義的詞彙（助動詞和通用動詞的詞形還原）
#     NEGATION_STOPWORDS = {"not", "do", "be", "have", "will",
#                           "can", "should", "get", "go", "say", "make", "allow", "taste"}

#     # --- 輔助函數：用於獲取形容詞的副詞修飾語（如 extremely, very） ---
#     def get_adv_modifier(token) -> str:
#         """從一個形容詞 token 獲取其主要的 advmod 修飾語。"""
#         adv_modifiers = []
#         for child in token.children:
#             if child.dep_ == "advmod" and child.pos_ == "ADV":
#                 lemma_lower = child.lemma_.lower()
#                 if lemma_lower not in STOPWORDS:
#                     adv_modifiers.append(lemma_lower)
#         return adv_modifiers[0] if adv_modifiers else ""

#     # --- Pattern 1: (ADV) + ADJ + NOUN (處理連續的形容詞 + 副詞修飾) ---
#     for i in range(len(doc)):
#         token = doc[i]

#         # 尋找一個有效名詞
#         if token.pos_ in VALID_NOUN_POS and token.lemma_.lower() not in STOPWORDS:
#             noun = token.lemma_.lower()
#             adjective_chain = []

#             # 向前迭代，尋找所有連在一起的有效修飾語
#             j = i - 1
#             while j >= 0 and doc[j].pos_ in VALID_MODIFIER_POS:
#                 current_token = doc[j]
#                 current_lemma = current_token.lemma_.lower()

#                 # 確保當前詞彙不是助動詞 (AUX) 或停用詞
#                 if current_token.pos_ != "AUX" and current_lemma not in STOPWORDS:

#                     # 1. 先插入形容詞 (因為是向前迭代，insert(0) 會把它放在鍊的最前面)
#                     adjective_chain.insert(0, current_lemma)

#                     # 2. [新增功能] 檢查這個形容詞是否有副詞修飾 (如 very -> sweet)
#                     adv = get_adv_modifier(current_token)
#                     if adv:
#                         # 如果有副詞，再插在形容詞前面
#                         # 最終順序範例: ["very", "sweet", "cake"]
#                         adjective_chain.insert(0, adv)

#                     j -= 1
#                 else:
#                     break

#             # 如果找到形容詞或修飾語，則組合片語
#             if adjective_chain:
#                 adj_part = "_".join(adjective_chain)
#                 phrases.append(f"{adj_part}_{noun}")

#     # --- Pattern 2: NOUN + 'be' + ADV + ADJ (使用依存關係解析) ---
#     for token in doc:
#         if token.pos_ == "ADJ":
#             adj = token.lemma_.lower()

#             if token.dep_ in ("acomp", "attr"):
#                 copular_verb = token.head

#                 for child in copular_verb.children:
#                     if child.dep_ == "nsubj" and child.pos_ in VALID_NOUN_POS:
#                         noun = child.lemma_.lower()

#                         is_negated = False
#                         for verb_child in copular_verb.children:
#                             if verb_child.dep_ == "neg" and verb_child.lemma_.lower() == "not":
#                                 is_negated = True
#                                 break

#                         adv = get_adv_modifier(token)

#                         if adj not in STOPWORDS:
#                             phrase_parts = []
#                             if is_negated:
#                                 phrase_parts.append("not")
#                             if adv:
#                                 phrase_parts.append(adv)
#                             phrase_parts.append(adj)

#                             # 如果名詞不是停用詞，就加進去
#                             if noun not in STOPWORDS:
#                                 phrase_parts.append(noun)
#                             else:
#                                 # 如果名詞是停用詞 (如 it)，但形容詞很長或有修飾，我們還是保留形容詞部分
#                                 # 或者加上一個通用後綴，避免單獨形容詞看起來很怪
#                                 pass

#                             # 確保片語夠長 (避免只提取出 'good')
#                             if len(phrase_parts) > 1:  # 例如 'very_sweet' 或 'not_good'
#                                 phrases.append("_".join(phrase_parts))

#     # --- Pattern 3: 動詞 + 副詞 / 動詞 + 否定 (捕捉行為評價) ---
#     for token in doc:
#         if token.pos_ == "VERB":
#             verb = token.lemma_.lower()

#             adv_modifier = ""
#             is_negated = False
#             noun_target = ""

#             for child in token.children:
#                 if child.dep_ == "advmod" and child.pos_ == "ADV" and child.lemma_.lower() not in STOPWORDS:
#                     adv_modifier = child.lemma_.lower()
#                 elif child.dep_ == "neg" and child.lemma_.lower() == "not":
#                     is_negated = True
#                 elif child.dep_ in ("nsubjpass", "dobj") and child.pos_ in VALID_NOUN_POS and child.lemma_.lower() not in STOPWORDS:
#                     noun_target = child.lemma_.lower()

#             # 只有當動詞被副詞或否定詞修飾時，且動詞本身不是通用動詞才提取
#             if (adv_modifier or is_negated) and verb not in STOPWORDS and verb not in NEGATION_STOPWORDS:

#                 phrase_parts = []

#                 if is_negated:
#                     phrase_parts.append("not")
#                 if adv_modifier:
#                     phrase_parts.append(adv_modifier)

#                 phrase_parts.append(verb)

#                 if noun_target:
#                     phrase_parts.append(noun_target)

#                 # 防止單詞 (如 'do' 或 'not') 被單獨提取
#                 if len(phrase_parts) == 1:
#                     continue

#                 phrases.append("_".join(phrase_parts))

#     # --- contraction artifact filter (正則過濾器) ---
#     contraction_artifact_1 = re.compile(r"^[a-z]+_t$")
#     contraction_artifact_2 = re.compile(
#         r"^(?:i|we|you|they|it|that|there|what|who|let)_(?:m|re|s|ll|ve|d)$")

#     # --- 最終強力過濾 ---
#     final_phrases = []
#     for phrase in phrases:
#         ph_low = phrase.lower().strip()

#         # Regex 過濾 (don_t, isn_t 等)
#         if contraction_artifact_1.fullmatch(ph_low) or contraction_artifact_2.fullmatch(ph_low):
#             continue

#         parts = ph_low.split('_')
#         lemmas = [p for p in parts if p]

#         # 單詞過濾 (do, not, be 等)
#         if len(lemmas) == 1:
#             single_word = lemmas[0]
#             if single_word in NEGATION_STOPWORDS or single_word in STOPWORDS:
#                 continue

#         final_phrases.append(ph_low)

#     return final_phrases

# def extract_adj_noun_phrases(text: str) -> List[str]:
#     """
#     從已清洗的英文句子中抽出 ADJ+NOUN 片語，並轉成 'adj_noun' 形式。
#     - Pattern 1 捕捉連續的形容詞+名詞結構 (例如: 'vietnamese_iced_coffee')。
#     - Pattern 2 使用依存關係捕捉 NOUN + 'be' + ADV + ADJ 結構，並包含副詞。
#     """
#     nlp = get_nlp()
#     doc = nlp(text)

#     phrases: List[str] = []

#     # --- 輔助函數：用於獲取形容詞的副詞修飾語（如 extremely, very） ---
#     def get_adv_modifier(token) -> str:
#         """從一個形容詞 token 獲取其主要的 advmod 修飾語。"""
#         # 尋找直接修飾這個形容詞的副詞 (advmod)
#         adv_modifiers = [
#             child.lemma_.lower()
#             for child in token.children
#             if child.dep_ == "advmod" and child.pos_ == "ADV"
#         ]
#         # 選擇最強烈或最靠近的副詞，這裡只取第一個
#         if adv_modifiers:
#             # 確保副詞不是通用停用詞
#             if adv_modifiers[0] not in STOPWORDS:
#                 return adv_modifiers[0]
#         return ""

#     # --- Pattern 1: ADJ + NOUN (處理連續的形容詞) ---
#     for i in range(len(doc)):
#         token = doc[i]

#         # 尋找一個名詞或專有名詞 (NOUN/PROPN)
#         if token.pos_ in {"NOUN", "PROPN"} and token.lemma_.lower() not in STOPWORDS:
#             noun = token.lemma_.lower()
#             adjective_chain = []

#             # 向前迭代，尋找所有連在一起的 ADJ
#             j = i - 1
#             while j >= 0 and doc[j].pos_ in {"ADJ", "NUM", "PROPN"} and doc[j].dep_ == "amod":
#                 # 將修飾語加入 chain 的最前端
#                 adjective_chain.insert(0, doc[j].lemma_.lower())
#                 j -= 1

#             # 如果找到形容詞或修飾語，則組合片語
#             if adjective_chain:
#                 # 確保形容詞鏈中沒有單一詞彙是 STOPWORDS
#                 if all(adj not in STOPWORDS for adj in adjective_chain):
#                     adj_part = "_".join(adjective_chain)
#                     phrases.append(f"{adj_part}_{noun}")

#     # --- Pattern 2: NOUN + 'be' + ADV + ADJ (使用依存關係解析，並包含副詞) ---
#     for token in doc:
#         # 從形容詞 (ADJ) 開始檢查
#         if token.pos_ == "ADJ":
#             adj = token.lemma_.lower()

#             # 1. 檢查形容詞是否為形容詞補語 (acomp) 或屬性 (attr)
#             if token.dep_ in ("acomp", "attr"):
#                 copular_verb = token.head

#                 # 2. 檢查這個動詞是否有一個名詞主語 (nsubj)
#                 for child in copular_verb.children:
#                     if child.dep_ == "nsubj" and child.pos_ in {"NOUN", "PROPN"}:
#                         noun = child.lemma_.lower()

#                         # 3. 獲取修飾這個 ADJ 的副詞 (如 extremely)
#                         adv = get_adv_modifier(token)

#                         # 4. 過濾並加入
#                         if adj not in STOPWORDS and noun not in STOPWORDS:
#                             phrase_parts = [adj, noun]

#                             # 如果有副詞，將其放在形容詞前
#                             if adv:
#                                 phrase_parts.insert(0, adv)

#                             phrases.append("_".join(phrase_parts))
#                             break

#     return phrases

# def extract_adj_noun_phrases(text: str) -> List[str]:
#     """
#     從已清洗的英文句子中抽出 ADJ+NOUN 片語，並轉成 'adj_noun' 形式。
#     這些 phrase 會用來跑 TF-IDF。
#     """
#     nlp = get_nlp()
#     doc = nlp(text)

#     phrases: List[str] = []

#     # Pattern 1: ADJ + NOUN (連在一起)
#     for i in range(len(doc) - 1):
#         t1, t2 = doc[i], doc[i + 1]
#         if t1.pos_ == "ADJ" and t2.pos_ in {"NOUN", "PROPN"}:
#             adj = t1.lemma_.lower()
#             noun = t2.lemma_.lower()
#             # 過濾 stopwords（你已將泛用名詞合併進 stopwords）
#             if (
#                 adj not in STOPWORDS
#                 and noun not in STOPWORDS
#             ):
#                 phrases.append(f"{adj}_{noun}")

#     # --- Pattern 2: NOUN + 'be' + ADJ (使用依存關係解析) ---
#     # 目標：捕捉 "Staff are often rude" -> "rude_staff"
#     for token in doc:
#         # 從形容詞 (ADJ) 開始檢查
#         if token.pos_ == "ADJ":
#             adj = token.lemma_.lower()

#             # 1. 檢查形容詞是否為形容詞補語 (acomp) 或屬性 (attr)，
#             #    這通常是 copular verb（如 'be'、'seem'）的補語
#             if token.dep_ in ("acomp", "attr"):
#                 # 取得 ADJ 的 Head（通常是動詞，如 'was', 'are'）
#                 copular_verb = token.head

#                 # 2. 檢查這個動詞是否有一個名詞主語 (nsubj)
#                 for child in copular_verb.children:
#                     if child.dep_ == "nsubj" and child.pos_ in {"NOUN", "PROPN"}:
#                         noun = child.lemma_.lower()

#                         # 3. 過濾並加入
#                         if adj not in STOPWORDS and noun not in STOPWORDS:
#                             phrases.append(f"{adj}_{noun}")
#                             # 找到主語後，結束對這個 ADJ 的檢查
#                             break

#     return phrases

# --------------------------------------------------
# 主流程：Step 4 + 5
# --------------------------------------------------


def analyze_aspect_phrases(input_path: str | Path, domain: Optional[str] = None) -> Path:
    """
    主入口（給 CLI 用的 lib call）：

    - 載入 reviews JSON (要有 text + rating)
    - 清洗 & phrase 抽取
    - 分 aspect + sentiment
    - 在每個 (aspect, sentiment) 子集合跑 phrase TF-IDF
    - 依 noun 聚合，輸出 CSV
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    place_name = input_path.stem
    # e.g. Holbox_reviews → Holbox
    if place_name.endswith("_reviews"):
        place_name = place_name[:-8]

    out_dir = OUTPUT_ROOT / place_name
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info(f"Analyzing aspect phrases for: {place_name}")
    logger.info(f"Input reviews: {input_path}")
    logger.info(f"Output dir   : {out_dir}")

    # ---- 載入 reviews：建議格式 list[dict]，有 "text" / "rating" ----
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # If domain wasn't provided, try to infer it from the input JSON schema
    if domain is None and isinstance(raw_data, dict):
        json_domain = raw_data.get("domain")
        if isinstance(json_domain, str) and json_domain.strip():
            domain = json_domain.strip()
            logger.info("Inferred domain from input JSON: %s", domain)

    # Initialize domain vocab/seeds (stopwords/protected_phrases/aspect_seeds.json)
    init_domain(domain)

    # Log input schema summary
    if isinstance(raw_data, dict):
        logger.info(
            "Loaded JSON object keys=%s",
            sorted(list(raw_data.keys()))[:20],
        )
        if "source" in raw_data:
            logger.info("Source      : %s", raw_data.get("source"))
        if "place_name" in raw_data:
            logger.info("Place name  : %s", raw_data.get("place_name"))
        if "domain" in raw_data:
            logger.info("Domain      : %s", raw_data.get("domain"))

    # 兼容：如果是 dict 並有 'reviews'，取其中的 list
    if isinstance(raw_data, dict) and "reviews" in raw_data:
        raw_data = raw_data["reviews"]

    if isinstance(raw_data, list):
        logger.info("Raw reviews count: %d", len(raw_data))

    records: List[Dict] = []
    total_items = 0
    skipped_no_text = 0
    skipped_no_cleaned = 0

    # For quick sanity check (avoid printing full text)
    sample_logged = 0

    for item in raw_data:
        total_items += 1
        if isinstance(item, str):
            text = item
            rating = None
            review_id = None
        elif isinstance(item, dict):
            review_id = item.get("review_id") or item.get("id")
            # Support multiple schemas (Google Maps unified reviews.json uses plain_text/raw_text + stars)
            text = (
                item.get("plain_text")
                or item.get("raw_text")
                or item.get("text")
                or item.get("reviewText")
                or item.get("content")
                or item.get("body")
                or item.get("review")
                or ""
            )
            rating = item.get("rating")
            if rating is None:
                rating = item.get("stars")
                if rating is None:
                    rating = item.get("star")
        else:
            continue

        if not text or not str(text).strip():
            skipped_no_text += 1
            continue

        cleaned = clean_review_text(str(text))
        if not cleaned:
            skipped_no_cleaned += 1
            continue

        if sample_logged < 5:
            logger.info(
                "Review ok #%d: id=%s, rating=%s, cleaned_len=%d",
                sample_logged + 1,
                review_id,
                rating,
                len(cleaned),
            )
            sample_logged += 1

        records.append(
            {
                "raw_text": text,
                "cleaned": cleaned,
                "rating": rating,
            }
        )

    logger.info(
        "Review ingest summary: total=%d, kept=%d, skipped_no_text=%d, skipped_no_cleaned=%d",
        total_items,
        len(records),
        skipped_no_text,
        skipped_no_cleaned,
    )

    # grouped_docs[(aspect, sentiment)] = list of "phrase phrase ..."
    grouped_docs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    # phrase_stats[(aspect, sentiment, phrase)] = count
    phrase_stats: Counter[Tuple[str, str, str]] = Counter()

    # ---- 對每則 review 做 phrase 抽取 + aspect + phrase-level sentiment ----
    for rec in records:
        rating = rec["rating"]
        review_level_sent = rating_to_sentiment(rating)  # 只當 fallback 用

        text_clean = rec["cleaned"]
        phrases = extract_adj_noun_phrases(text_clean)
        if not phrases:
            continue

        # aspect_sent_map[(aspect, sentiment)] = [phrase, ...]（同一則 review 內）
        aspect_sent_map: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for ph in phrases:
            # 1) 先決定屬於哪個 aspect
            aspect = assign_aspect(ph.replace("_", " "))
            if not aspect:
                continue

            # 2) 再決定正/負（phrase 級別）
            sent = phrase_sentiment(ph, aspect)

            # 3) 如果看不出正負，才用整體 rating 當備援
            if sent is None:
                if review_level_sent is None:
                    continue  # 完全看不出來，就忽略這個 phrase
                sent = review_level_sent

            key = (aspect, sent)
            aspect_sent_map[key].append(ph)

        # 這則 review 的所有 (aspect, sentiment) → 合併成 document
        for (aspect, sent), ph_list in aspect_sent_map.items():
            if not ph_list:
                continue
            key = (aspect, sent)
            doc_str = " ".join(ph_list)
            grouped_docs[key].append(doc_str)
            for ph in ph_list:
                phrase_stats[(aspect, sent, ph)] += 1

    if not grouped_docs:
        logger.warning("No aspect phrases extracted; check seeds / data.")
        return out_dir

    # ---- 在每個 (aspect, sentiment) 子集合裡跑 phrase TF-IDF ----

    rows_tfidf: List[Dict] = []

    for (aspect, sentiment), docs in grouped_docs.items():
        # 先過濾掉空字串 / 全空白的文件
        docs_clean = [d for d in docs if isinstance(d, str) and d.strip()]
        n_docs = len(docs_clean)
        if n_docs < 2:
            logger.debug(
                f"Skip TF-IDF for aspect={aspect}, sentiment={sentiment}: not enough non-empty docs ({n_docs})"
            )
            continue

        # 動態設定 ngram_range
        if n_docs < 15:
            ngram_range = (1, 1)
        elif n_docs < 50:
            ngram_range = (1, 2)
        else:
            ngram_range = (1, 3)

        # 動態設定 min_df
        if n_docs <= 10:
            min_df = 1
        elif n_docs <= 50:
            min_df = 2
        else:
            min_df = max(2, int(0.01 * n_docs))

        logger.info(
            f"TF-IDF for aspect={aspect}, sentiment={sentiment}: n_docs={n_docs}, ngram_range={ngram_range}, min_df={min_df}"
        )

        vec = TfidfVectorizer(
            token_pattern=r"(?u)\b\w+\b",  # 允許含底線的 token
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words=None,
        )

        try:
            X = vec.fit_transform(docs_clean)
        except ValueError as e:
            logger.warning(
                "Skipping TF-IDF for aspect=%s, sentiment=%s (n_docs=%d, ngram_range=%s, min_df=%s) due to error: %s",
                aspect,
                sentiment,
                n_docs,
                ngram_range,
                min_df,
                e,
            )
            continue

        terms = vec.get_feature_names_out()
        scores = np.asarray(X.sum(axis=0)).ravel()

        order = np.argsort(scores)[::-1]
        for idx in order:
            ph = terms[idx]
            score = float(scores[idx])
            freq = int(phrase_stats.get((aspect, sentiment, ph), 0))
            rows_tfidf.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "phrase": ph,
                    "tfidf_sum": score,
                    "freq": freq,
                }
            )

    if not rows_tfidf:
        logger.warning("TF-IDF produced no rows; maybe docs too few.")
        return out_dir

    tfidf_df = pd.DataFrame(rows_tfidf)
    tfidf_path = out_dir / f"aspect_phrases_tfidf_{run_ts}.csv"
    tfidf_df.to_csv(tfidf_path, index=False)
    logger.info(f"Saved phrase TF-IDF to {tfidf_path}")

    # ---- 聚合：依 noun 整理形容詞列表（給人看 & AI summary） ----

    agg_rows: List[Dict] = []
    for (aspect, sentiment), sub_df in tfidf_df.groupby(["aspect", "sentiment"]):
        # target_stats[noun] = Counter(adj -> count)
        target_adj_counts: Dict[str, Counter] = defaultdict(Counter)

        for _, row in sub_df.iterrows():
            ph = row["phrase"]  # e.g. "bland_soup"
            freq = int(row.get("freq", 0))
            parts = ph.split("_", 1)
            if len(parts) != 2:
                continue
            adj, noun = parts[0], parts[1]
            target_adj_counts[noun][adj] += max(freq, 1)

        for noun, adj_counter in target_adj_counts.items():
            total_mentions = int(sum(adj_counter.values()))
            top_adjs = [f"{adj}({cnt})" for adj,
                        cnt in adj_counter.most_common(5)]
            agg_rows.append(
                {
                    "aspect": aspect,
                    "sentiment": sentiment,
                    "target_noun": noun,
                    "total_mentions": total_mentions,
                    "top_adjectives": ", ".join(top_adjs),
                }
            )

    if agg_rows:
        agg_df = pd.DataFrame(agg_rows).sort_values(
            ["aspect", "sentiment", "total_mentions"],
            ascending=[True, True, False],
        )
        agg_path = out_dir / f"aspect_targets_aggregated_{run_ts}.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregated targets to {agg_path}")
    else:
        logger.warning(
            "No aggregated targets produced; check phrase extraction & TF-IDF.")

    # 這裡的 out_dir 裡就會有：
    # - aspect_phrases_tfidf.csv（給你看 phrase 排名）
    # - aspect_targets_aggregated.csv（給使用者 / AI summary 用）
    return out_dir


def save_top5_keywords_csv(tfidf_df: pd.DataFrame, output_dir: Path, run_ts: str) -> None:
    """
    從 TF-IDF 結果中篩選出各面向 (Aspect) 與情感 (Sentiment) 下，
    TF-IDF Sum 最高的前 5 個關鍵字，並儲存為 CSV。
    """
    if tfidf_df.empty:
        return

    # 1. 排序：先依 Aspect, Sentiment 分組，再依 TF-IDF Sum (降序) 和 Freq (降序) 排序
    sorted_df = tfidf_df.sort_values(
        by=['aspect', 'sentiment', 'tfidf_sum', 'freq'],
        ascending=[True, True, False, False]
    )

    # 2. 分組並取前 5 名
    top5_df = sorted_df.groupby(['aspect', 'sentiment']).head(5)

    # 3. 整理欄位
    columns_order = ['aspect', 'sentiment', 'phrase', 'tfidf_sum', 'freq']
    existing_cols = [c for c in columns_order if c in top5_df.columns]
    final_df = top5_df[existing_cols]

    # 4. 存檔
    filename = f"top5_keywords_tfidf_{run_ts}.csv"
    output_path = output_dir / filename
    try:
        final_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved Top 5 keywords to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to save Top 5 keywords CSV: {e}")


def save_aspect_analysis_results(tfidf_df: pd.DataFrame, agg_df: pd.DataFrame, out_dir: Path) -> None:
    """
    統一負責將分析結果儲存為 CSV 檔案。
    會產出：
    1. aspect_phrases_tfidf_{ts}.csv (完整關鍵字清單)
    2. top5_keywords_tfidf_{ts}.csv (各面向 Top 5)
    3. aspect_targets_aggregated_{ts}.csv (依名詞聚合的結果)
    """
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # 統一產生一個 timestamp，確保所有檔案的時間戳記一致
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 儲存完整 TF-IDF 表
    if not tfidf_df.empty:
        tfidf_path = out_dir / f"aspect_phrases_tfidf_{run_ts}.csv"
        try:
            tfidf_df.to_csv(tfidf_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved aspect phrases TF-IDF to {tfidf_path}")

            # [新增] 同時儲存 Top 5 報告
            save_top5_keywords_csv(tfidf_df, out_dir, run_ts)

        except Exception as e:
            logger.error(f"Failed to save TF-IDF CSV: {e}")
    else:
        logger.warning("TF-IDF DataFrame is empty, skipping save.")

    # 2. 儲存 Aggregated Targets 表
    if not agg_df.empty:
        agg_path = out_dir / f"aspect_targets_aggregated_{run_ts}.csv"
        try:
            agg_df.to_csv(agg_path, index=False, encoding='utf-8-sig')
            logger.info(f"Saved aggregated targets to {agg_path}")
        except Exception as e:
            logger.error(f"Failed to save Aggregated CSV: {e}")
    else:
        logger.warning("Aggregated DataFrame is empty, skipping save.")


if __name__ == "__main__":
    # 手動測試用：python -m spotlite.analysis.aspect_phrases path/to/Holbox_reviews.json [domain]
    import sys
    # Ensure logs are visible when running as a script/module
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        force=True,
    )
    logger.setLevel(logging.INFO)
    if len(sys.argv) < 2:
        print(
            "Usage: python -m spotlite.analysis.aspect_phrases path/to/reviews.json [domain]")
        raise SystemExit(1)
    reviews_path = sys.argv[1]
    domain_arg = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_aspect_phrases(reviews_path, domain=domain_arg)
