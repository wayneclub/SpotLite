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

import json
import re
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

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

# 一些在餐廳情境下太泛用、通常不想當作 target 的 generic 名詞
GENERIC_NOUNS: set[str] = {
    "food",
    "meal",
    "meals",
    "dish",
    "dishes",
    "plate",
    "plates",
    "stuff",
    "thing",
    "things",
}

# 全域 opinion hints：補強 phrase_sentiment，避免過度依賴星等
GLOBAL_POS_SEEDS: List[str] = [
    "amazing",
    "excellent",
    "great",
    "awesome",
    "fantastic",
    "worth",
    "worth it",
    "worth the wait",
    "worth_wait",
    "delicious",
    "tasty",
    "fresh",
    "perfect",
]

GLOBAL_NEG_SEEDS: List[str] = [
    "terrible",
    "awful",
    "horrible",
    "bad",
    "bland",
    "overcooked",
    "cold",
    "soggy",
    "greasy",
    "slow",
    "rude",
    "long wait",
    "overpriced",
    "expensive",
    "noisy",
    "crowded",
]


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

    global STOPWORDS, PROTECTED_PHRASES, ASPECT_SEEDS, ASPECT_SEEDS_RAW, _ASPECT_CENTERS

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
    對單一 phrase (e.g. 'bland_soup') 做情緒判斷：
      - 回傳 "pos" / "neg" / None（中立或不確定）

    使用該 aspect 下的 seeds_pos / seeds_neg 做 cosine similarity 比較：
      score = max_sim_pos - max_sim_neg
      > threshold  → pos
      < -threshold → neg
      其餘         → None
    """
    if aspect not in ASPECT_SEEDS_RAW:
        return None

    seeds_obj = ASPECT_SEEDS_RAW.get(aspect, {})
    pos_seeds = list({str(s) for s in (seeds_obj.get(
        "seeds_pos", []) or [])} | set(GLOBAL_POS_SEEDS))
    neg_seeds = list({str(s) for s in (seeds_obj.get(
        "seeds_neg", []) or [])} | set(GLOBAL_NEG_SEEDS))

    if not pos_seeds and not neg_seeds:
        return None

    text = phrase.replace("_", " ")  # "fresh_fish" -> "fresh fish"
    model = get_aspect_model()
    v = model.encode(text, convert_to_tensor=True)

    sims_pos: List[float] = []
    sims_neg: List[float] = []

    if pos_seeds:
        emb_pos = model.encode(pos_seeds, convert_to_tensor=True)
        sims_pos = util.cos_sim(v, emb_pos)[0].tolist()

    if neg_seeds:
        emb_neg = model.encode(neg_seeds, convert_to_tensor=True)
        sims_neg = util.cos_sim(v, emb_neg)[0].tolist()

    best_pos = max(sims_pos) if sims_pos else -1.0
    best_neg = max(sims_neg) if sims_neg else -1.0

    score = best_pos - best_neg  # >0 越偏正向，<0 越偏負向

    if score > threshold:
        return "pos"
    elif score < -threshold:
        return "neg"
    else:
        return None

# --------------------------------------------------
# Phrase extraction (ADJ + NOUN)
# --------------------------------------------------


_NLP = None


def get_nlp():
    """
    準備 spaCy 英文模型：en_core_web_sm
    需先執行：
      python -m spacy download en_core_web_sm
    """
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extract_adj_noun_phrases(text: str) -> List[str]:
    """
    從已清洗的英文句子中抽出 ADJ+NOUN 片語，並轉成 'adj_noun' 形式。
    這些 phrase 會用來跑 TF-IDF。
    """
    nlp = get_nlp()
    doc = nlp(text)

    phrases: List[str] = []

    # Pattern 1: ADJ + NOUN (連在一起)
    for i in range(len(doc) - 1):
        t1, t2 = doc[i], doc[i + 1]
        if t1.pos_ == "ADJ" and t2.pos_ in {"NOUN", "PROPN"}:
            adj = t1.lemma_.lower()
            noun = t2.lemma_.lower()
            # 過濾過於泛用的名詞（例如 food/meal/dish），但會保留 protected_phrases 產生的複合詞
            if (
                adj not in STOPWORDS
                and noun not in STOPWORDS
                and noun not in GENERIC_NOUNS
            ):
                phrases.append(f"{adj}_{noun}")

    # Pattern 2: NOUN + 'be' + ADJ
    for i in range(len(doc) - 2):
        t1, t2, t3 = doc[i], doc[i + 1], doc[i + 2]
        if t1.pos_ in {"NOUN", "PROPN"} and t2.lemma_ == "be" and t3.pos_ == "ADJ":
            noun = t1.lemma_.lower()
            adj = t3.lemma_.lower()
            if (
                adj not in STOPWORDS
                and noun not in STOPWORDS
                and noun not in GENERIC_NOUNS
            ):
                phrases.append(f"{adj}_{noun}")

    return phrases

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

    # 根據指定 domain（或 configs.json 裡的 default_domain）初始化詞彙設定
    init_domain(domain)

    place_name = input_path.stem
    # e.g. Holbox_reviews → Holbox
    if place_name.endswith("_reviews"):
        place_name = place_name[:-8]

    out_dir = OUTPUT_ROOT / place_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing aspect phrases for: {place_name}")
    logger.info(f"Input reviews: {input_path}")
    logger.info(f"Output dir   : {out_dir}")

    # ---- 載入 reviews：建議格式 list[dict]，有 "text" / "rating" ----
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # 兼容：如果是 dict 並有 'reviews'，取其中的 list
    if isinstance(raw_data, dict) and "reviews" in raw_data:
        raw_data = raw_data["reviews"]

    records: List[Dict] = []
    for item in raw_data:
        if isinstance(item, str):
            text = item
            rating = None
        elif isinstance(item, dict):
            text = (
                item.get("text")
                or item.get("reviewText")
                or item.get("content")
                or item.get("body")
                or item.get("review")
                or ""
            )
            rating = item.get("rating")
        else:
            continue

        if not text or not text.strip():
            continue

        cleaned = clean_review_text(text)
        if not cleaned:
            continue

        records.append(
            {
                "raw_text": text,
                "cleaned": cleaned,
                "rating": rating,
            }
        )

    logger.info(f"Loaded & cleaned {len(records)} reviews")

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
        if len(docs_clean) < 2:
            # 資料太少，跳過
            logger.debug(
                f"Skip TF-IDF for aspect={aspect}, sentiment={sentiment}: not enough non-empty docs ({len(docs_clean)})"
            )
            continue

        vec = TfidfVectorizer(
            ngram_range=(1, 1),   # phrase 已經是單一 token（例如 'fresh_fish'）
            min_df=1,             # phrase 至少出現在 1 則 review
            stop_words=None,
            token_pattern=r"(?u)\b\w+\b",  # 允許含底線的 token
        )

        try:
            X = vec.fit_transform(docs_clean)
        except ValueError as e:
            # 例如：After pruning, no terms remain. Try a lower min_df or a higher max_df.
            logger.warning(
                f"Skipping TF-IDF for aspect={aspect}, sentiment={sentiment} due to error: {e}"
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
    tfidf_path = out_dir / "aspect_phrases_tfidf.csv"
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
        agg_path = out_dir / "aspect_targets_aggregated.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregated targets to {agg_path}")
    else:
        logger.warning(
            "No aggregated targets produced; check phrase extraction & TF-IDF.")

    # 這裡的 out_dir 裡就會有：
    # - aspect_phrases_tfidf.csv（給你看 phrase 排名）
    # - aspect_targets_aggregated.csv（給使用者 / AI summary 用）
    return out_dir


if __name__ == "__main__":
    # 手動測試用：python -m spotlite.analysis.aspect_phrases path/to/Holbox_reviews.json [domain]
    import sys
    if len(sys.argv) < 2:
        print(
            "Usage: python -m spotlite.analysis.aspect_phrases path/to/reviews.json [domain]")
        raise SystemExit(1)
    reviews_path = sys.argv[1]
    domain_arg = sys.argv[2] if len(sys.argv) > 2 else None
    analyze_aspect_phrases(reviews_path, domain=domain_arg)
