"""
Aspect Keyword Analyzer (Ultimate Edition)
------------------------------------------
功能集大成：
1. NER 實體過濾 (自動封殺地名/人名)
2. Protected Phrases 強制提取與保護
3. Stopwords 智慧過濾 (支援底線連接詞)
4. 多層次 Aspect 分類 (強勢形容詞 > 種子詞 > 核心名詞 AI > 整句 AI)
5. 情感分析 (支援雙重否定、情境規則)
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOMAINS_ROOT = PROJECT_ROOT / "spotlite/config/domains"

GENERAL_CFG = load_config("configs.json")
PATH_CFG = GENERAL_CFG.get("paths", {})
OUTPUT_ROOT = Path(PATH_CFG.get(
    "aspect_phrases_output_root", "outputs/aspect_phrases"))
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Global Resources
# --------------------------------------------------
_nlp_instance = None
_aspect_model = None


def get_nlp():
    global _nlp_instance
    if _nlp_instance is None:
        try:
            _nlp_instance = spacy.load("en_core_web_lg")
        except OSError:
            logger.warning("en_core_web_lg not found, using sm.")
            _nlp_instance = spacy.load("en_core_web_sm")
    return _nlp_instance


def get_aspect_model():
    global _aspect_model
    if _aspect_model is None:
        _aspect_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _aspect_model

# --------------------------------------------------
# Main Class
# --------------------------------------------------


class AspectKeywordAnalyzer:
    def __init__(self, domain: str = "restaurant"):
        self.domain = domain
        self.raw_records: List[Dict] = []
        self.processed_records: List[Dict] = []
        self.tfidf_df: pd.DataFrame = pd.DataFrame()
        self.input_stem = "analysis"

        self.stopwords: Set[str] = set(ENGLISH_STOP_WORDS)
        self.protected_phrases: Dict[str, str] = {}
        self.protected_tokens: Set[str] = set()

        self.aspect_seeds: Dict[str, List[str]] = {}
        self.aspect_seeds_raw: Dict[str, Dict[str, List[str]]] = {}
        self.global_pos_seeds: List[str] = []
        self.global_neg_seeds: List[str] = []

        self.aspect_embeddings = {}
        self.sentiment_embeddings = {}

        self._load_domain_resources(domain)
        self._precompute_seed_embeddings()

    def _load_domain_resources(self, domain: str):
        logger.info(f"Loading resources for domain: {domain}")
        base = DOMAINS_ROOT / domain

        # 1. Stopwords (支援 list 或 dict 格式)
        sw_path = base / "stopwords.json"
        if sw_path.exists():
            try:
                with open(sw_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.stopwords.update(data)
                        count = len(data)
                    elif isinstance(data, dict):
                        sw_list = data.get("stopwords")
                        if sw_list is None:
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

        # 2. Protected Phrases
        pp_path = base / "protected_phrases.json"
        if pp_path.exists():
            try:
                with open(pp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.protected_phrases = data.get("protected_phrases", {})
                    self.protected_tokens = set(
                        self.protected_phrases.values())
                logger.info(
                    f"Loaded {len(self.protected_phrases)} protected phrases.")
            except Exception as e:
                logger.error(f"Failed to load protected phrases: {e}")

        # 3. Aspect Seeds
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
                logger.info(f"Loaded seeds.")
            except Exception as e:
                logger.error(f"Failed to load seeds: {e}")

    def _precompute_seed_embeddings(self):
        if not self.aspect_seeds:
            return
        logger.info("Caching embeddings...")
        model = get_aspect_model()

        for aspect, seeds in self.aspect_seeds.items():
            if seeds:
                self.aspect_embeddings[aspect] = model.encode(
                    seeds, convert_to_tensor=True)

        for aspect, content in self.aspect_seeds_raw.items():
            if aspect == "global_sentiment":
                continue
            pos = list(
                set((content.get("seeds_pos") or []) + self.global_pos_seeds))
            neg = list(
                set((content.get("seeds_neg") or []) + self.global_neg_seeds))

            self.sentiment_embeddings[aspect] = {
                "pos": model.encode(pos, convert_to_tensor=True) if pos else None,
                "neg": model.encode(neg, convert_to_tensor=True) if neg else None
            }

    def _apply_protected_phrases(self, text: str) -> str:
        if not self.protected_phrases:
            return text
        processed = text
        for phrase, replacement in sorted(self.protected_phrases.items(), key=lambda x: len(x[0]), reverse=True):
            if phrase in processed:
                processed = processed.replace(phrase, replacement)
        return processed

    def _extract_phrases_internal(self, text: str) -> List[str]:
        nlp = get_nlp()
        doc = nlp(text)
        phrases: List[str] = []

        VALID_NOUN = {"NOUN", "PROPN"}
        VALID_MOD = {"ADJ", "NUM", "PROPN"}

        GARBAGE = {"don", "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
                   "won", "wouldn", "shouldn", "couldn", "mustn", "t", "s", "re", "ve", "m", "ll", "d"}
        NEGATION = {"not", "do", "be", "have", "will", "can",
                    "should", "get", "go", "say", "make", "allow", "taste"}

        CHECK_SET = self.stopwords.union(GARBAGE).union(NEGATION)

        # [保留] 副詞白名單
        KEEP_ADVERBS = {"so", "too", "very", "really",
                        "highly", "extremely", "quite", "super"}

        # [黑名單] 自動封殺實體
        entity_blacklist = set()
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "LOC"):  # 人名, 地名
                entity_blacklist.add(ent.text.lower())
                for part in ent.text.split():
                    entity_blacklist.add(part.lower())

        # [Pattern 0] 強制提取保護詞
        protected_set = set(self.protected_phrases.values())
        for token in doc:
            if token.text in protected_set:
                phrases.append(token.text)

        def get_adv(token) -> str:
            advs = []
            for c in token.children:
                if c.dep_ == "advmod" and c.pos_ == "ADV":
                    lemma = c.lemma_.lower()
                    if lemma in KEEP_ADVERBS or lemma not in CHECK_SET:
                        advs.append(lemma)
            return advs[0] if advs else ""

        # Pattern 1: (ADV) + ADJ + NOUN
        for i in range(len(doc)):
            token = doc[i]
            if token.text.lower() in entity_blacklist:
                continue

            # 避免重複提取保護詞
            if token.text in protected_set:
                continue

            if token.pos_ in VALID_NOUN and token.lemma_.lower() not in CHECK_SET:
                noun = token.lemma_.lower()
                adj_chain = []
                j = i - 1
                while j >= 0 and doc[j].pos_ in VALID_MOD:
                    curr = doc[j]
                    lem = curr.lemma_.lower()
                    if curr.text.lower() in entity_blacklist:
                        break
                    if curr.text in protected_set:
                        break

                    if curr.pos_ != "AUX" and lem not in CHECK_SET:
                        adj_chain.insert(0, lem)
                        adv = get_adv(curr)
                        if adv:
                            adj_chain.insert(0, adv)
                        j -= 1
                    else:
                        break
                if adj_chain:
                    phrases.append(f"{'_'.join(adj_chain)}_{noun}")

        # Pattern 2: NOUN + be + (ADV) + ADJ
        for token in doc:
            if token.pos_ == "ADJ":
                adj = token.lemma_.lower()
                has_adv = any(
                    c.lemma_.lower() in KEEP_ADVERBS for c in token.children if c.dep_ == "advmod")
                if not has_adv and adj in CHECK_SET:
                    continue

                if token.dep_ in ("acomp", "attr"):
                    copular = token.head
                    noun = next((c.lemma_.lower(
                    ) for c in copular.children if c.dep_ == "nsubj" and c.pos_ in VALID_NOUN), None)

                    noun_token_text = next((c.text.lower(
                    ) for c in copular.children if c.dep_ == "nsubj" and c.pos_ in VALID_NOUN), "")
                    if noun and (noun in entity_blacklist or noun_token_text in entity_blacklist):
                        continue

                    is_neg = any(c.dep_ == "neg" for c in copular.children)
                    adv = get_adv(token)

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

        # Pattern 3: (ADV) + VERB + (ADV/ADJ) + (NOUN)
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
                    elif c.dep_ == "advmod" and c.pos_ == "ADV" and c.i < token.i:
                        lemma = c.lemma_.lower()
                        if lemma in KEEP_ADVERBS or lemma not in CHECK_SET:
                            pre_adv = lemma
                    elif c.dep_ in ("advmod", "acomp") and c.i > token.i:
                        lemma = c.lemma_.lower()
                        if lemma not in CHECK_SET:
                            post_attr = lemma
                    elif c.dep_ in ("nsubjpass", "dobj") and c.pos_ in VALID_NOUN:
                        if c.text.lower() in entity_blacklist:
                            continue
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
        seen = set()

        for ph in phrases:
            ph_low = ph.lower().strip()
            if ph_low in seen:
                continue
            if rgx1.fullmatch(ph_low) or rgx2.fullmatch(ph_low):
                continue
            parts = ph_low.split('_')
            if len(parts) == 1 and parts[0] in CHECK_SET and parts[0] not in protected_set:
                continue
            final.append(ph_low)
            seen.add(ph_low)

        return final

    def _assign_aspect(self, text: str) -> Optional[str]:
        if not self.aspect_seeds or not self.aspect_embeddings:
            return None

        text_lower = text.lower()
        parts = text_lower.split('_')

        # ---------------------------------------------------------
        # 1. Stopwords 過濾 (取代原本的 ABSOLUTE_IGNORE)
        # ---------------------------------------------------------
        # 檢查 A: 完全匹配 (e.g. "not_sure" 在 stopwords 裡)
        if text_lower in self.stopwords:
            return None

        # 檢查 B: 部分匹配 (e.g. "past_week" -> 包含 "week")
        # 注意：我們需要一個 "豁免清單"，避免誤殺像 "dirty_spoon" 這種
        # 因為 "spoon" 在 stopwords 裡，但我們想保留 "dirty_spoon"

        # 定義豁免形容詞 (如果出現這些詞，就算包含 stopword 也不殺)
        KEEP_TRIGGERS = {
            "dirty", "filthy", "stained", "unclean", "grimy", "smelly",  # 環境
            "broken", "chipped", "cracked",                             # 破損
            "no", "missing", "lack", "ask", "need",                     # 服務
            # 特殊專名 (雖然通常已被 protected)
            "michelin", "texas", "nashville"
        }

        # 掃描每個字根
        for part in parts:
            if part in self.stopwords:
                # 關鍵邏輯：雖然發現了 stopword (如 spoon)，但如果有豁免詞 (如 dirty)，就放行
                # 如果沒有豁免詞，就殺掉
                if not any(t in text_lower for t in KEEP_TRIGGERS):
                    return None

        # ---------------------------------------------------------
        # 2. 強勢形容詞優先 (Dominant Priority)
        # ---------------------------------------------------------
        PRIORITY_MAP = {
            "price": ["affordable", "cheap", "expensive", "price", "cost", "bill", "value", "worth", "overpriced", "pricy", "reasonable", "rip-off", "check", "dollar", "buck", "money"],
            "waiting_time": ["wait", "queue", "line", "slow", "fast", "rush", "forever", "minute", "hour", "long time"],
            "service": ["rude", "friendly", "polite", "helpful", "staff", "waiter", "waitress", "manager", "server", "host", "hostess"]
        }
        for aspect, keywords in PRIORITY_MAP.items():
            if any(k in text_lower for k in keywords):
                if "fast food" in text_lower and aspect == "waiting_time":
                    continue
                return aspect

        # ---------------------------------------------------------
        # 3. 種子詞精確匹配 (Exact Seed Match)
        # ---------------------------------------------------------
        for aspect, seeds in self.aspect_seeds.items():
            for seed in seeds:
                if len(seed) > 2 and seed in text_lower:
                    return aspect

        # ---------------------------------------------------------
        # 4. 核心名詞 AI 比對 (Head Noun Embedding)
        # ---------------------------------------------------------
        model = get_aspect_model()
        head_word = parts[-1]

        if len(head_word) > 2:
            v_head = model.encode(head_word, convert_to_tensor=True)
            best_head_aspect, max_head_score = None, -1.0

            for aspect, emb_seeds in self.aspect_embeddings.items():
                sims = util.cos_sim(v_head, emb_seeds)
                score = float(sims.max())
                if score > max_head_score:
                    max_head_score = score
                    best_head_aspect = aspect

            if max_head_score >= 0.35:
                return best_head_aspect

        # ---------------------------------------------------------
        # 5. 整句 AI 比對 (Full Phrase Embedding)
        # ---------------------------------------------------------
        v_full = model.encode(text.replace("_", " "), convert_to_tensor=True)
        best_aspect, max_score = None, -1.0

        for aspect, emb_seeds in self.aspect_embeddings.items():
            sims = util.cos_sim(v_full, emb_seeds)
            score = float(sims.max())
            if score > max_score:
                max_score = score
                best_aspect = aspect

        return best_aspect if max_score >= 0.28 else None

    def _phrase_sentiment(self, phrase: str, aspect: str) -> Optional[str]:
        if aspect not in self.sentiment_embeddings:
            return None
        parts = phrase.lower().split('_')
        head = parts[-1]
        text_lower = phrase.lower().replace("_", " ")

        seeds_obj = self.aspect_seeds_raw.get(aspect, {})
        aspect_neg_seeds = seeds_obj.get(
            "seeds_neg", []) if isinstance(seeds_obj, dict) else []
        ALL_NEG_ADJS = set(self.global_neg_seeds).union(aspect_neg_seeds)

        if any(k in parts for k in {"not", "no", "never", "didn", "don", "wont", "cant"}):
            if head in ALL_NEG_ADJS or any(adj in text_lower for adj in ALL_NEG_ADJS):
                return "pos"
            return "neg"

        TIME_UNITS = {"hour", "minute", "hr", "min",
                      "forever", "eternity", "age", "long"}
        if ("spend" in parts or "spent" in parts) and any(unit in text_lower for unit in TIME_UNITS):
            return "neg"
        if "limited" in parts:
            return "neg"

        if any(k in parts for k in {"too", "overly", "excessively"}):
            aspect_pos_seeds = seeds_obj.get(
                "seeds_pos", []) if isinstance(seeds_obj, dict) else []
            ALL_POS_ADJS = set(self.global_pos_seeds).union(aspect_pos_seeds)
            if head in ALL_POS_ADJS:
                return "pos"
            return "neg"

        embeds = self.sentiment_embeddings[aspect]
        if embeds["pos"] is None and embeds["neg"] is None:
            return None
        model = get_aspect_model()
        v = model.encode(text_lower, convert_to_tensor=True)
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

    def load_data(self, input_path: str | Path) -> 'AspectKeywordAnalyzer':
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        self.input_stem = input_path.stem.replace("_reviews", "")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.raw_records = raw_data.get(
            "reviews") or raw_data.get("data") or raw_data
        logger.info(f"Loaded {len(self.raw_records)} reviews.")
        return self

    def extract_phrases(self) -> 'AspectKeywordAnalyzer':
        if not self.raw_records:
            return self
        logger.info("Extracting phrases...")
        processed = []
        for item in self.raw_records:
            text = (item.get("text") or item.get("plain_text") or "")
            if not text or not str(text).strip():
                continue
            text_lower = str(text).lower()
            text_protected = self._apply_protected_phrases(text_lower)
            cleaned = clean_en(text_protected)
            phrases = self._extract_phrases_internal(cleaned)

            for ph in phrases:
                aspect = self._assign_aspect(ph)
                if not aspect:
                    continue
                sent = self._phrase_sentiment(ph, aspect)
                if sent is None:
                    continue
                processed.append(
                    {"aspect": aspect, "sentiment": sent, "phrase": ph})

        self.processed_records = processed
        logger.info(f"Extraction complete. Found {len(processed)} phrases.")
        return self

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

    def get_top_keywords(self, top_n: int = 5) -> pd.DataFrame:
        if self.tfidf_df.empty:
            return pd.DataFrame()
        # [NEW] 過濾低頻雜訊
        filtered_df = self.tfidf_df[self.tfidf_df['freq'] > 1]
        if filtered_df.empty:
            return self.tfidf_df.head(top_n)  # Fallback

        return filtered_df.sort_values(
            by=['aspect', 'sentiment', 'tfidf_sum', 'freq'],
            ascending=[True, True, False, False]
        ).groupby(['aspect', 'sentiment']).head(top_n)

    def save_results(self):
        if self.tfidf_df.empty:
            return
        out_dir = OUTPUT_ROOT / self.input_stem
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        top5 = self.get_top_keywords(5)
        top5.to_csv(
            out_dir / f"top5_keywords_{ts}.csv", index=False, encoding='utf-8-sig')
        self.tfidf_df.to_csv(
            out_dir / f"aspect_phrases_tfidf_{ts}.csv", index=False, encoding='utf-8-sig')
        logger.info(f"Results saved to {out_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python aspect_analyzer.py path/to/reviews.json [domain]")
        sys.exit(1)
    AspectKeywordAnalyzer(sys.argv[2] if len(sys.argv) > 2 else "restaurant").load_data(
        sys.argv[1]).extract_phrases().compute_tfidf().save_results()
    print("Done.")
