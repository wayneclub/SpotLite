"""
Aspect Keyword Analyzer (Fixed Tensor Boolean Error)
----------------------------------------------------
修正了 PyTorch Tensor 在布林判斷時的 RuntimeError。
完全封裝、自動快取、規則優先的情感分析模組。
"""

import json
import re
import logging
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer, util
import spacy

# --------------------------------------------------
# Config Loading
# --------------------------------------------------
try:
    from spotlite.config.config import load_config
    from spotlite.analysis.utils_text import clean_en
except ImportError:
    logging.warning("Spotlite modules not found. Using defaults.")
    def load_config(path): return {}
    def clean_en(text): return text.lower().strip().replace('\n', ' ')

logger = logging.getLogger(__name__)

# 設定專案路徑
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAINS_ROOT = PROJECT_ROOT / "spotlite/config/domains"

# 輸出路徑
GENERAL_CFG = load_config("configs.json")
PATH_CFG = GENERAL_CFG.get("paths", {})
OUTPUT_ROOT = Path(PATH_CFG.get(
    "aspect_phrases_output_root", "outputs/aspect_phrases"))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Global Resources (Lazy Loading)
# --------------------------------------------------
_nlp_instance = None
_aspect_model = None


def get_nlp():
    global _nlp_instance
    if _nlp_instance is None:
        try:
            _nlp_instance = spacy.load("en_core_web_lg")
        except OSError:
            logger.warning(
                "en_core_web_lg not found, falling back to en_core_web_sm")
            _nlp_instance = spacy.load("en_core_web_sm")
    return _nlp_instance


def get_aspect_model():
    global _aspect_model
    if _aspect_model is None:
        _aspect_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _aspect_model

# --------------------------------------------------
# Main Class: AspectKeywordAnalyzer
# --------------------------------------------------


class AspectKeywordAnalyzer:
    def __init__(self, domain: str = "restaurant"):
        self.domain = domain
        self.raw_records: List[Dict] = []
        self.processed_records: List[Dict] = []
        self.tfidf_df: pd.DataFrame = pd.DataFrame()
        self.input_stem = "analysis"

        # 領域資源
        self.stopwords: Set[str] = set(ENGLISH_STOP_WORDS)
        self.aspect_seeds: Dict[str, List[str]] = {}
        self.aspect_seeds_raw: Dict[str, Dict[str, List[str]]] = {}
        self.global_pos_seeds: List[str] = []
        self.global_neg_seeds: List[str] = []

        # 快取 (Embeddings Cache)
        self.aspect_embeddings = {}
        self.sentiment_embeddings = {}

        # 初始化流程
        self._load_domain_resources(domain)
        self._precompute_seed_embeddings()

    # ------------------------------------------------------------------
    # Step 0: Resources & Caching
    # ------------------------------------------------------------------
    def _load_domain_resources(self, domain: str):
        logger.info(f"Loading resources for domain: {domain}")
        base = DOMAINS_ROOT / domain

        # 1. Stopwords (修正讀取邏輯)
        sw_path = base / "stopwords.json"
        if sw_path.exists():
            try:
                with open(sw_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                    # [修正點] 判斷資料結構
                    if isinstance(data, list):
                        # 情況 A: 檔案內容是 ["a", "b", ...]
                        self.stopwords.update(data)
                        count = len(data)
                    elif isinstance(data, dict):
                        # 情況 B: 檔案內容是 {"stopwords": ["a", "b", ...]}
                        # 嘗試取得 "stopwords" 欄位，如果沒有則取 values() 的總和
                        sw_list = data.get("stopwords")
                        if sw_list is None:
                            # 萬一 key 不叫 stopwords，把所有 values 展平
                            sw_list = []
                            for v in data.values():
                                if isinstance(v, list):
                                    sw_list.extend(v)

                        if sw_list:
                            self.stopwords.update(sw_list)
                            count = len(sw_list)
                        else:
                            count = 0

                logger.info(f"Loaded {count} custom stopwords.")
            except Exception as e:
                logger.error(f"Failed to load stopwords: {e}")
        else:
            logger.warning(f"Stopwords file missing: {sw_path}")

        # 2. Aspect Seeds (保持不變)
        seeds_path = base / "aspect_seeds.json"
        if seeds_path.exists():
            try:
                with open(seeds_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.aspect_seeds_raw = data
                    for aspect, content in data.items():
                        if aspect == "global_sentiment":
                            continue
                        if isinstance(content, list):
                            self.aspect_seeds[aspect] = content
                        else:
                            pos = content.get("seeds_pos", []) or []
                            neg = content.get("seeds_neg", []) or []
                            self.aspect_seeds[aspect] = list(set(pos + neg))

                    if "global_sentiment" in data:
                        self.global_pos_seeds = data["global_sentiment"].get(
                            "pos", [])
                        self.global_neg_seeds = data["global_sentiment"].get(
                            "neg", [])
                logger.info(
                    f"Loaded seeds for aspects: {list(self.aspect_seeds.keys())}")
            except Exception as e:
                logger.error(f"Failed to load seeds: {e}")

    def _precompute_seed_embeddings(self):
        """一次性計算並快取種子詞向量"""
        if not self.aspect_seeds:
            return

        logger.info("Pre-computing seed embeddings (Caching)...")
        model = get_aspect_model()

        # 1. Aspect Embeddings
        for aspect, seeds in self.aspect_seeds.items():
            if seeds:
                self.aspect_embeddings[aspect] = model.encode(
                    seeds, convert_to_tensor=True)

        # 2. Sentiment Embeddings
        for aspect, content in self.aspect_seeds_raw.items():
            if aspect == "global_sentiment":
                continue

            # 使用 get 安全獲取，避免 Key Error
            pos_seeds_raw = content.get(
                "seeds_pos", []) if isinstance(content, dict) else []
            neg_seeds_raw = content.get(
                "seeds_neg", []) if isinstance(content, dict) else []

            pos = list(set(pos_seeds_raw + self.global_pos_seeds))
            neg = list(set(neg_seeds_raw + self.global_neg_seeds))

            self.sentiment_embeddings[aspect] = {
                "pos": model.encode(pos, convert_to_tensor=True) if pos else None,
                "neg": model.encode(neg, convert_to_tensor=True) if neg else None
            }
        logger.info("Embeddings cached.")

    # ------------------------------------------------------------------
    # Step 1: Load Data
    # ------------------------------------------------------------------
    def load_data(self, input_path: str | Path) -> 'AspectKeywordAnalyzer':
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)

        self.input_stem = input_path.stem
        if self.input_stem.endswith("_reviews"):
            self.input_stem = self.input_stem[:-8]

        logger.info(f"Loading data from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        if isinstance(raw_data, dict):
            self.raw_records = raw_data.get(
                "reviews") or raw_data.get("data") or []
        elif isinstance(raw_data, list):
            self.raw_records = raw_data

        logger.info(f"Loaded {len(self.raw_records)} reviews.")
        return self

    # ------------------------------------------------------------------
    # Step 2: Extraction & Analysis (Internal Methods)
    # ------------------------------------------------------------------
    def _extract_phrases_internal(self, text: str) -> List[str]:
        nlp = get_nlp()
        doc = nlp(text)
        phrases: List[str] = []

        VALID_NOUN = {"NOUN", "PROPN"}
        VALID_MOD = {"ADJ", "NUM", "PROPN"}

        # 1. 定義垃圾詞與停用詞集合
        GARBAGE = {"don", "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
                   "won", "wouldn", "shouldn", "couldn", "mustn", "t", "s", "re", "ve", "m", "ll", "d"}
        NEGATION = {"not", "do", "be", "have", "will", "can",
                    "should", "get", "go", "say", "make", "allow", "taste"}

        CHECK_SET = self.stopwords.union(GARBAGE).union(NEGATION)

        # 2. [新增] 程度副詞白名單 (即使在 stopwords 也要保留)
        KEEP_ADVERBS = {"so", "too", "very", "really",
                        "highly", "extremely", "quite", "super"}

        # 3. [新增] 人名過濾 (NER) - 解決 Cody ChesnuTT 問題
        # 找出句子中所有被標記為 PERSON 的實體
        person_blacklist = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # 將全名 (Cody ChesnuTT) 和單字 (Cody, ChesnuTT) 都加入黑名單
                person_blacklist.add(ent.text.lower())
                for part in ent.text.split():
                    person_blacklist.add(part.lower())

        # 輔助函數：獲取副詞 (包含白名單邏輯)
        def get_adv(token) -> str:
            advs = []
            for c in token.children:
                if c.dep_ == "advmod" and c.pos_ == "ADV":
                    lemma = c.lemma_.lower()
                    # 邏輯修正：如果是白名單副詞，或者不在停用詞表中 -> 保留
                    if lemma in KEEP_ADVERBS or lemma not in CHECK_SET:
                        advs.append(lemma)
            return advs[0] if advs else ""

        # --- Pattern 1: (ADV) + ADJ + NOUN ---
        for i in range(len(doc)):
            token = doc[i]

            # [新增] 人名檢查：如果是人名，直接跳過
            if token.text.lower() in person_blacklist:
                continue

            if token.pos_ in VALID_NOUN and token.lemma_.lower() not in CHECK_SET:
                noun = token.lemma_.lower()
                adj_chain = []
                j = i - 1
                while j >= 0 and doc[j].pos_ in VALID_MOD:
                    curr = doc[j]
                    lem = curr.lemma_.lower()

                    # 檢查修飾語是否為人名
                    if curr.text.lower() in person_blacklist:
                        break  # 中斷鍊接

                    if curr.pos_ != "AUX" and lem not in CHECK_SET:
                        adj_chain.insert(0, lem)
                        adv = get_adv(curr)
                        if adv:
                            adj_chain.insert(0, adv)  # 這裡會成功插入 "so"
                        j -= 1
                    else:
                        break

                if adj_chain:
                    phrases.append(f"{'_'.join(adj_chain)}_{noun}")

        # --- Pattern 2: NOUN + be + (ADV) + ADJ ---
        for token in doc:
            if token.pos_ == "ADJ":
                adj = token.lemma_.lower()

                # [新增] 對形容詞本身做停用詞檢查 (除非有副詞修飾)
                has_adv = any(
                    c.lemma_.lower() in KEEP_ADVERBS for c in token.children if c.dep_ == "advmod")
                if not has_adv and adj in CHECK_SET:
                    continue

                if token.dep_ in ("acomp", "attr"):
                    copular = token.head
                    noun = next((c.lemma_.lower(
                    ) for c in copular.children if c.dep_ == "nsubj" and c.pos_ in VALID_NOUN), None)

                    # 檢查主詞是否為人名
                    if noun and noun in person_blacklist:
                        continue

                    is_neg = any(c.dep_ == "neg" for c in copular.children)
                    adv = get_adv(token)  # 這裡會抓到 "so"

                    parts = []
                    if is_neg:
                        parts.append("not")
                    if adv:
                        parts.append(adv)
                    parts.append(adj)
                    if noun and noun not in CHECK_SET:
                        parts.append(noun)

                    if len(parts) > 1:
                        phrases.append("_".join(parts))

        # --- Pattern 3: (ADV) + VERB + (ADV/ADJ) + (NOUN) ---
        for token in doc:
            if token.pos_ == "VERB":
                verb = token.lemma_.lower()
                if verb in CHECK_SET:
                    continue

                pre_adv, post_attr, noun_target = "", "", ""
                is_neg = False

                for c in token.children:
                    if c.dep_ == "neg":
                        is_neg = True
                    # 前置副詞 (Pre-adv)
                    elif c.dep_ == "advmod" and c.pos_ == "ADV" and c.i < token.i:
                        lemma = c.lemma_.lower()
                        # 同樣應用白名單邏輯
                        if lemma in KEEP_ADVERBS or lemma not in CHECK_SET:
                            pre_adv = lemma
                    # 後置修飾 (Post-attr)
                    elif c.dep_ in ("advmod", "acomp") and c.i > token.i:
                        lemma = c.lemma_.lower()
                        if lemma not in CHECK_SET:
                            post_attr = lemma
                    # 受詞 (Object)
                    elif c.dep_ in ("nsubjpass", "dobj") and c.pos_ in VALID_NOUN:
                        if c.lemma_.lower() not in CHECK_SET:
                            noun_target = c.lemma_.lower()

                parts = []
                if is_neg:
                    parts.append("not")
                if pre_adv:
                    parts.append(pre_adv)
                parts.append(verb)
                if post_attr:
                    parts.append(post_attr)
                if noun_target:
                    parts.append(noun_target)

                if len(parts) > 1:
                    phrases.append("_".join(parts))

        # Final Filter
        final = []
        rgx1 = re.compile(r"^[a-z]+_t$")
        rgx2 = re.compile(
            r"^(?:i|we|you|they|it|that|there|what|who|let)_(?:m|re|s|ll|ve|d)$")

        for ph in phrases:
            ph_low = ph.lower().strip()
            if rgx1.fullmatch(ph_low) or rgx2.fullmatch(ph_low):
                continue

            parts = ph_low.split('_')

            # 單詞過濾：如果是單詞且在 CHECK_SET 中 -> 丟棄
            # 注意：這裡不需額外檢查人名，因為前面已經擋掉了，且這裡主要擋停用詞
            if len(parts) == 1 and parts[0] in CHECK_SET:
                continue

            final.append(ph_low)

        return final

    def _assign_aspect(self, text: str) -> Optional[str]:
        if not self.aspect_embeddings:
            return None
        model = get_aspect_model()
        v = model.encode(text.replace("_", " "), convert_to_tensor=True)

        best, max_score = None, -1.0

        for aspect, emb_seeds in self.aspect_embeddings.items():
            score = float(util.cos_sim(v, emb_seeds).max())
            if score > max_score:
                max_score = score
                best = aspect

        return best if max_score >= 0.25 else None

    def _phrase_sentiment(self, phrase: str, aspect: str) -> Optional[str]:
        # 快取中沒有這個 aspect (可能 seeds 沒載入)
        if aspect not in self.sentiment_embeddings:
            return None

        parts = phrase.lower().split('_')
        head = parts[-1]

        # 1. Rules: Too/Overly/Negation
        if any(k in parts for k in {"too", "overly", "excessively"}):
            NEG_SENSORY = {"sweet", "salty", "sour", "bitter", "spicy",
                           "dry", "hard", "cold", "greasy", "expensive", "loud", "slow"}
            if head in NEG_SENSORY:
                return "neg"
            return "pos"

        if any(k in parts for k in {"not", "no", "never", "didn", "don"}):
            NEG_ADJS = {"bad", "terrible", "awful", "gross",
                        "rude", "slow", "dirty", "expensive"}
            return "pos" if head in NEG_ADJS else "neg"

        # 2. AI (Cached) - FIX IS HERE
        embeds = self.sentiment_embeddings[aspect]
        # 修正：明確檢查是否為 None，因為 Tensor 不能直接用 if not 判斷
        if embeds["pos"] is None and embeds["neg"] is None:
            return None

        model = get_aspect_model()
        v = model.encode(phrase.replace("_", " "), convert_to_tensor=True)

        # 使用 is not None 檢查 Tensor
        s_pos = float(util.cos_sim(v, embeds["pos"]).max(
        )) if embeds["pos"] is not None else -1.0
        s_neg = float(util.cos_sim(v, embeds["neg"]).max(
        )) if embeds["neg"] is not None else -1.0

        score = s_pos - s_neg
        if score > 0.05:
            return "pos"
        elif score < -0.05:
            return "neg"
        return None

    # ------------------------------------------------------------------
    # Step 2: Orchestration
    # ------------------------------------------------------------------
    def extract_phrases(self) -> 'AspectKeywordAnalyzer':
        if not self.raw_records:
            logger.warning("No records to process.")
            return self

        logger.info("Extracting phrases...")
        processed = []

        for item in self.raw_records:
            text = (item.get("text") or item.get("plain_text")
                    or item.get("reviewText") or "")
            if not text or not str(text).strip():
                continue

            cleaned = clean_en(str(text))
            phrases = self._extract_phrases_internal(cleaned)

            for ph in phrases:
                aspect = self._assign_aspect(ph)
                if not aspect:
                    continue

                sent = self._phrase_sentiment(ph, aspect)
                if sent is None:
                    continue

                processed.append({
                    "aspect": aspect,
                    "sentiment": sent,
                    "phrase": ph
                })

        self.processed_records = processed
        logger.info(f"Extraction complete. Found {len(processed)} phrases.")
        return self

    # ------------------------------------------------------------------
    # Step 3: TF-IDF
    # ------------------------------------------------------------------
    def compute_tfidf(self) -> 'AspectKeywordAnalyzer':
        if not self.processed_records:
            return self

        grouped_docs = defaultdict(list)
        phrase_counts = Counter()

        for rec in self.processed_records:
            key = (rec['aspect'], rec['sentiment'])
            phrase_counts[(rec['aspect'], rec['sentiment'],
                           rec['phrase'])] += 1
            grouped_docs[key].append(rec['phrase'])

        rows = []
        for (aspect, sentiment), ph_list in grouped_docs.items():
            doc_str = " ".join(ph_list)
            # 簡單防護：如果文件太短不跑 TF-IDF，直接用頻率
            if len(ph_list) < 3:
                for ph in set(ph_list):
                    rows.append({
                        "aspect": aspect, "sentiment": sentiment, "phrase": ph,
                        "tfidf_sum": float(phrase_counts[(aspect, sentiment, ph)]),
                        "freq": phrase_counts[(aspect, sentiment, ph)]
                    })
                continue

            vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", min_df=1)
            try:
                X = vec.fit_transform([doc_str])
                terms = vec.get_feature_names_out()
                scores = X.toarray()[0]

                for ph, score in zip(terms, scores):
                    rows.append({
                        "aspect": aspect, "sentiment": sentiment, "phrase": ph,
                        "tfidf_sum": float(score),
                        "freq": phrase_counts[(aspect, sentiment, ph)]
                    })
            except ValueError:
                continue

        self.tfidf_df = pd.DataFrame(rows)
        return self

    # ------------------------------------------------------------------
    # Step 4: Output
    # ------------------------------------------------------------------
    def get_top_keywords(self, top_n: int = 5) -> pd.DataFrame:
        """
        獲取各面向 Top N 關鍵字。
        改進點：
        1. 使用混合分數 (TF-IDF * sqrt(Freq)) 排序，優先展示熱門且重要的詞。
        2. 自動過濾 Freq = 1 的詞，去除偶發雜訊 (如錯字、人名)。
        """
        if self.tfidf_df.empty:
            return pd.DataFrame()

        # 1. 複製一份以免影響原資料
        df_scored = self.tfidf_df.copy()

        # 2. [關鍵過濾] 只保留出現超過 1 次的詞
        # 這能瞬間過濾掉 90% 的錯字 (bad_ecause) 和罕見人名 (Cody)
        df_scored = df_scored[df_scored['freq'] > 1]

        if df_scored.empty:
            return pd.DataFrame()

        # 3. [混合分數] Score = TF-IDF * sqrt(Frequency)
        # 讓高頻詞 (Popularity) 的權重提升
        # 需要 import numpy as np
        df_scored['hybrid_score'] = df_scored['tfidf_sum'] * \
            np.sqrt(df_scored['freq'])

        # 4. 排序並取 Top N
        return df_scored.sort_values(
            by=['aspect', 'sentiment', 'hybrid_score'],
            ascending=[True, True, False]
        ).groupby(['aspect', 'sentiment']).head(top_n)

    def save_results(self):
        if self.tfidf_df.empty:
            logger.warning("No results to save.")
            return

        out_dir = OUTPUT_ROOT / self.input_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. Top 5
        top5 = self.get_top_keywords(5)
        top5.to_csv(
            out_dir / f"top5_keywords_{ts}.csv", index=False, encoding='utf-8-sig')

        # 2. Full
        self.tfidf_df.to_csv(
            out_dir / f"aspect_phrases_tfidf_{ts}.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python aspect_analyzer.py path/to/reviews.json [domain]")
        sys.exit(1)

    fpath = sys.argv[1]
    domain_arg = sys.argv[2] if len(sys.argv) > 2 else "restaurant"

    AspectKeywordAnalyzer(domain_arg).load_data(
        fpath).extract_phrases().compute_tfidf().save_results()
    print("Done.")
