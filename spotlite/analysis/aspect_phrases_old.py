"""
Aspect Keyword Analyzer (Ultimate Edition v13 - Final Aggregation Fix)
------------------------------------------
版本更新 (v13)：
1. [CRITICAL FIX] 通用名詞雙重降級：在 normalize_noun 中，對像 'food', 'meal', 'dish' 這樣的通用名詞進行二次降級，讓它們優先被更有描述性的詞替換。
2. [CRITICAL FIX] 聚合顯示強化：在 aggregate_group 中，強制優先選擇帶有形容詞的多詞片語 (如 delicious_appetizer)，即使單詞 'appetizer' 頻率更高。
3. 包含所有先前版本的修正。
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
        self.global_sentiment_terms: Set[str] = set()

        self.aspect_embeddings = {}
        self.sentiment_embeddings = {}

        self._load_domain_resources(domain)
        self._precompute_seed_embeddings()

    def _load_domain_resources(self, domain: str):
        logger.info(f"Loading resources for domain: {domain}")
        base = DOMAINS_ROOT / domain
        nlp = get_nlp()

        def to_lemma_set(word_list):
            lemmas = set()
            for w in word_list:
                doc = nlp(w)
                lemma_form = "_".join([t.lemma_.lower() for t in doc])
                lemmas.add(lemma_form)
            return lemmas

        # 1. Stopwords
        sw_path = base / "stopwords.json"
        if sw_path.exists():
            try:
                with open(sw_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    raw_list = []
                    if isinstance(data, list):
                        raw_list = data
                    elif isinstance(data, dict):
                        raw_list = data.get("stopwords")
                        if raw_list is None:
                            raw_list = []
                            for v in data.values():
                                if isinstance(v, list):
                                    raw_list.extend(v)

                    if raw_list:
                        self.stopwords.update(to_lemma_set(raw_list))
                        count = len(raw_list)
                    else:
                        count = 0
                logger.info(
                    f"Loaded {count} custom stopwords (auto-lemmatized).")
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

                        all_seeds = []
                        if isinstance(content, list):
                            all_seeds = content
                        else:
                            kw = content.get("keywords", []) or []
                            pos = content.get("seeds_pos", []) or []
                            neg = content.get("seeds_neg", []) or []
                            all_seeds = kw + pos + neg

                        self.aspect_seeds[aspect] = list(
                            to_lemma_set(all_seeds))

                    if "global_sentiment" in data:
                        self.global_pos_seeds = list(to_lemma_set(
                            data["global_sentiment"].get("pos", [])))
                        self.global_neg_seeds = list(to_lemma_set(
                            data["global_sentiment"].get("neg", [])))

                        self.global_sentiment_terms.update(
                            self.global_pos_seeds)
                        self.global_sentiment_terms.update(
                            self.global_neg_seeds)

                logger.info(f"Loaded seeds (auto-lemmatized).")
            except Exception as e:
                logger.error(f"Failed to load seeds: {e}")

    def _precompute_seed_embeddings(self):
        if not self.aspect_seeds:
            return
        logger.info("Caching embeddings...")
        model = get_aspect_model()
        nlp = get_nlp()

        def lemmatize_list(l):
            return ["_".join([t.lemma_.lower() for t in nlp(w)]) for w in l]

        for aspect, seeds in self.aspect_seeds.items():
            if seeds:
                self.aspect_embeddings[aspect] = model.encode(
                    seeds, convert_to_tensor=True)

        for aspect, content in self.aspect_seeds_raw.items():
            if aspect == "global_sentiment":
                continue

            raw_pos = (content.get("seeds_pos") or []
                       ) if isinstance(content, dict) else []
            raw_neg = (content.get("seeds_neg") or []
                       ) if isinstance(content, dict) else []

            pos = list(set(lemmatize_list(raw_pos) + self.global_pos_seeds))
            neg = list(set(lemmatize_list(raw_neg) + self.global_neg_seeds))

            self.sentiment_embeddings[aspect] = {
                "pos": model.encode(pos, convert_to_tensor=True) if pos else None,
                "neg": model.encode(neg, convert_to_tensor=True) if neg else None
            }

    def _apply_protected_phrases(self, text: str) -> str:
        if not self.protected_phrases:
            return text
        processed = text
        sorted_phrases = sorted(
            self.protected_phrases.items(), key=lambda x: len(x[0]), reverse=True)

        for phrase, replacement in sorted_phrases:
            safe_phrase = re.escape(phrase).replace(r"\ ", r"\s+")
            pattern = rf"\b{safe_phrase}(?:s|es)?\b"
            processed = re.sub(pattern, replacement,
                               processed, flags=re.IGNORECASE)

        return processed

    def _extract_phrases_internal(self, text: str) -> List[str]:
        nlp = get_nlp()
        doc = nlp(text)
        phrases: List[str] = []

        VALID_NOUN = {"NOUN", "PROPN"}
        # [FIX] 包含動詞分詞 VBN/VBG (e.g., seasoned, fried)
        VALID_MOD = {"ADJ", "NUM", "PROPN", "VERB"}

        GARBAGE = {"don", "isn", "aren", "wasn", "weren", "haven", "hasn", "hadn",
                   "won", "wouldn", "shouldn", "couldn", "mustn", "t", "s", "re", "ve", "m", "ll", "d"}

        NEGATION = {"not", "do", "be", "have", "will", "can",
                    "should", "get", "go", "say", "make", "allow", "taste",
                    "visit", "love", "like"}

        CHECK_SET = self.stopwords.union(GARBAGE).union(NEGATION)
        KEEP_ADVERBS = {"so", "too", "very", "really",
                        "highly", "extremely", "quite", "super", "pretty", "honestly", "truly"}

        entity_blacklist = set()
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "GPE", "LOC"):
                entity_blacklist.add(ent.text.lower())
                for part in ent.text.split():
                    entity_blacklist.add(part.lower())

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

        # Pattern 1: (NEG?) + (ADV) + ADJ + NOUN
        for i in range(len(doc)):
            token = doc[i]
            if token.text.lower() in entity_blacklist:
                continue

            if token.pos_ in VALID_NOUN and token.lemma_.lower() not in CHECK_SET:
                noun = token.lemma_.lower()
                adj_chain = []

                remote_negation = ""
                head_verb = token.head
                if head_verb.pos_ == "VERB":
                    for child in head_verb.children:
                        if child.dep_ == "neg" or child.lemma_.lower() in {"never", "no", "not"}:
                            remote_negation = child.lemma_.lower()
                            break

                j = i - 1
                while j >= 0 and doc[j].pos_ in VALID_MOD:
                    curr = doc[j]
                    lem = curr.lemma_.lower()

                    if curr.text.lower() in entity_blacklist or lem in CHECK_SET:
                        break

                    if curr.pos_ != "AUX":
                        # 檢查動詞分詞標籤，確保是修飾語
                        if curr.pos_ == "VERB" and curr.tag_ not in ("VBN", "VBG"):
                            break

                        adj_chain.insert(0, lem)
                        adv = get_adv(curr)
                        if adv:
                            adj_chain.insert(0, adv)
                        j -= 1
                    else:
                        break

                if adj_chain:
                    phrase_body = f"{'_'.join(adj_chain)}_{noun}"
                    if remote_negation and "not" not in adj_chain and "never" not in adj_chain:
                        phrases.append(f"{remote_negation}_{phrase_body}")
                    else:
                        phrases.append(phrase_body)

        # Pattern 2: NOUN + be + (ADV) + ADJ/VBN/VBG
        for token in doc:
            if token.pos_ == "ADJ" or (token.pos_ == "VERB" and token.tag_ in ("VBN", "VBG")):
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

                    is_neg = any(c.dep_ == "neg" or c.lemma_ in {
                                 "never", "not"} for c in copular.children)
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

        # Pattern 3
        for token in doc:
            if token.pos_ == "VERB":
                verb = token.lemma_.lower()
                if verb in CHECK_SET:
                    continue
                pre_adv, post_attr, noun_target = "", "", ""
                is_neg = False
                for c in token.children:
                    if c.dep_ == "neg" or c.lemma_ in {"never", "not"}:
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

        # Pattern 4
        CRITICAL_HYGIENE = {"gloves", "glove",
                            "mask", "masks", "hand", "hands"}
        for i in range(len(doc)):
            token = doc[i]
            if token.text.lower() in CRITICAL_HYGIENE:
                phrase_parts = []
                head = token.head

                if head.pos_ == "VERB" and head.lemma_.lower() in {"change", "wash", "wear", "use"}:
                    is_neg = False
                    for child in head.children:
                        if child.dep_ == "neg":
                            is_neg = True

                    if is_neg:
                        phrase_parts.append("not")
                    phrase_parts.append(head.lemma_.lower())
                    phrase_parts.append(token.lemma_.lower())

                elif head.pos_ == "VERB" and head.dep_ == "pobj":
                    prep = head.head
                    if prep.text.lower() == "without":
                        phrase_parts.append("without")
                        phrase_parts.append(head.lemma_.lower())
                        phrase_parts.append(token.lemma_.lower())

                if phrase_parts:
                    phrases.append("_".join(phrase_parts))

        # Final Filter & Deduplication
        final = []
        rgx1 = re.compile(r"^[a-z]+_t$")
        rgx2 = re.compile(
            r"^(?:i|we|you|they|it|that|there|what|who|let)_(?:m|re|s|ll|ve|d)$")
        seen = set()

        for ph in phrases:
            ph_low = ph.lower().strip()

            # 疊字去重 (love_love -> love)
            ph_low = re.sub(r'\b(\w+)(?:_\1)+\b', r'\1', ph_low)

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

        nlp = get_nlp()
        text_lower = text.lower()
        parts = text_lower.split('_')

        parts_lemma = []
        for p in parts:
            p_lem = nlp(p)[0].lemma_.lower()
            parts_lemma.append(p_lem)
        text_lemma = "_".join(parts_lemma)

        # [FIX] 0. 獨立情感詞過濾
        if len(parts_lemma) == 1 and parts_lemma[0] in self.global_sentiment_terms:
            return None

        # 1. Stopwords 過濾
        if text_lemma in self.stopwords:
            return None

        KEEP_TRIGGERS = {
            "dirty", "filthy", "stained", "unclean", "grimy", "smelly",
            "broken", "chipped", "cracked",
            "no", "missing", "lack", "ask", "need",
            "michelin", "texas", "nashville",
            "without", "change", "wear"
        }

        for part in parts_lemma:
            if part in self.stopwords:
                if not any(t in text_lower for t in KEEP_TRIGGERS):
                    return None

        # 2. 強勢形容詞優先
        PRIORITY_MAP = {
            "price": ["affordable", "cheap", "expensive", "price", "cost", "bill", "value", "worth", "overpriced", "pricy", "reasonable", "rip-off", "check", "dollar", "buck", "money"],
            "waiting_time": ["wait", "queue", "line", "slow", "fast", "rush", "forever", "minute", "hour", "long time"],
            "service": ["rude", "friendly", "polite", "helpful", "staff", "waiter", "waitress", "manager", "server", "host", "hostess", "glove", "mask", "hygiene"]
        }
        for aspect, keywords in PRIORITY_MAP.items():
            if any(k in text_lower for k in keywords):
                if "fast food" in text_lower and aspect == "waiting_time":
                    continue
                return aspect

        # 3. 種子詞精確匹配 (含 keywords)
        for aspect, seeds in self.aspect_seeds.items():
            for seed in seeds:
                if len(seed) > 2 and seed in text_lemma:
                    return aspect

        # 4. 核心名詞 AI 比對
        model = get_aspect_model()
        head_word = parts_lemma[-1]

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

        # 5. 整句 AI 比對
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
        raw_neg = (seeds_obj.get("seeds_neg", [])
                   if isinstance(seeds_obj, dict) else [])

        nlp = get_nlp()
        neg_lemma_set = set(["_".join([t.lemma_.lower()
                            for t in nlp(w)]) for w in raw_neg])
        ALL_NEG_ADJS = set(self.global_neg_seeds).union(neg_lemma_set)

        # 否定詞反轉邏輯
        if any(k in parts for k in {"not", "no", "never", "didn", "don", "wont", "cant", "without"}):
            head_lemma = nlp(head)[0].lemma_.lower()

            if head_lemma in ALL_NEG_ADJS or any(adj in text_lower for adj in ALL_NEG_ADJS):
                return "pos"
            return "neg"

        TIME_UNITS = {"hour", "minute", "hr", "min",
                      "forever", "eternity", "age", "long"}
        if ("spend" in parts or "spent" in parts) and any(unit in text_lower for unit in TIME_UNITS):
            return "neg"
        if "limited" in parts:
            return "neg"

        if any(k in parts for k in {"too", "overly", "excessively"}):
            raw_pos = (seeds_obj.get("seeds_pos", [])
                       if isinstance(seeds_obj, dict) else [])
            pos_lemma_set = set(["_".join([t.lemma_.lower()
                                for t in nlp(w)]) for w in raw_pos])
            ALL_POS_ADJS = set(self.global_pos_seeds).union(pos_lemma_set)

            head_lemma = nlp(head)[0].lemma_.lower()
            if head_lemma in ALL_POS_ADJS:
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

        nlp = get_nlp()
        df = self.tfidf_df.copy()

        FIXED_PHRASES = {
            'soft_shell_crab', 'blue_crab', 'king_crab', 'snow_crab',
            'spicy_tuna', 'spicy_salmon', 'handroll', 'crispy_rice',
            'happy_hour', 'tasting_menu', 'omakase', 'baked_crab', 'unagi_shrimp_tempura'
        }

        GENERIC_HEADS = {
            'place', 'spot', 'location', 'area', 'restaurant', 'shop', 'store',
            'joint', 'thing', 'way', 'option', 'choice', 'selection', 'experience',
            'dish', 'item', 'one', 'bit', 'love', 'like'
        }

        # [NEW FIX] 額外的通用名詞集合 (用於降級 Head Noun)
        SECONDARY_GENERIC_HEADS = {
            'food', 'meal', 'cuisine', 'appetizer', 'entree', 'dessert', 'snack'}

        SYNONYM_MAP = {
            'hour': 'time', 'minute': 'time', 'min': 'time', 'hr': 'time',
            'second': 'time', 'sec': 'time', 'moment': 'time', 'while': 'time',
            'handrolls': 'handroll', 'tacos': 'taco', 'fries': 'fry', 'chips': 'chip',
            'drinks': 'drink', 'cocktails': 'cocktail', 'beers': 'beer', 'rolls': 'roll',
            'venue': 'place',
            'staff': 'service', 'waiter': 'service', 'server': 'service', 'employee': 'service',
            'atmosphere': 'vibe', 'ambiance': 'vibe'
        }

        PROTECTED_VALUES = set(self.protected_phrases.values())
        FIXED_PHRASES.update(PROTECTED_VALUES)

        def normalize_noun(phrase):
            parts = phrase.split('_')

            # [FIX] Step 0: 專門矯正動詞片語/固定名詞的 Head Noun
            is_action_phrase = len(
                parts) >= 2 and parts[-2] in {'change', 'wash', 'wear'}
            is_hygiene_noun = parts[-1] in {'glove', 'mask'}

            if is_action_phrase or is_hygiene_noun:
                if parts[0] in {'not', 'without'}:
                    return "_".join(parts[1:])
                return "_".join(parts)

            # Step 1: 絕對豁免檢查
            if phrase in FIXED_PHRASES:
                return phrase

            for fixed in FIXED_PHRASES:
                if phrase.endswith(f"_{fixed}"):
                    return fixed

            # Step 2: 略過通用結尾 (e.g. recommend_place -> recommend)
            if len(parts) > 1 and parts[-1] in GENERIC_HEADS:
                raw_noun = parts[-2]
            else:
                raw_noun = parts[-1]

            # [CRITICAL FIX] Step 3: 通用名詞雙重降級
            if raw_noun in SECONDARY_GENERIC_HEADS and len(parts) > 1 and parts[0] not in SECONDARY_GENERIC_HEADS:
                # 如果 Head Noun 是通用詞，且前面有修飾語（如 delicious），則將 Head Noun 設為修飾詞 (delicious_appetizer -> delicious)
                # 但我們不能返回形容詞，所以我們返回 '修飾語+名詞' 的完整片語來確保它獨立成群
                return phrase  # 返回完整片語，阻止其與單詞 'appetizer' 歸併

            # Step 4: 同義詞與原形歸併
            if raw_noun in SYNONYM_MAP:
                return SYNONYM_MAP[raw_noun]

            lemma = nlp(raw_noun)[0].lemma_.lower()
            if lemma in SYNONYM_MAP:
                return SYNONYM_MAP[lemma]

            if lemma == raw_noun.lower() and lemma.endswith('s') and not lemma.endswith('ss'):
                singular = lemma[:-1]
                if singular in SYNONYM_MAP:
                    return SYNONYM_MAP[singular]
                return singular

            return lemma

        df['head_noun'] = df['phrase'].apply(normalize_noun)

        def aggregate_group(group):
            total_tfidf = group['tfidf_sum'].sum()
            total_freq = group['freq'].sum()

            candidates = group.sort_values(
                by=['freq', 'tfidf_sum'], ascending=False)

            # [CRITICAL FIX] 優先級 1: 檢查是否有帶有形容詞的長片語
            # 必須包含形容詞/副詞，且不是單詞
            descriptive_phrases = candidates[
                (candidates['phrase'].str.contains('_')) &
                (~candidates['phrase'].apply(lambda x: x in FIXED_PHRASES)) &
                (~candidates['phrase'].apply(
                    lambda x: x in SECONDARY_GENERIC_HEADS))
            ].sort_values(by=['freq', 'tfidf_sum'], ascending=False)

            if not descriptive_phrases.empty:
                best_phrase = descriptive_phrases.iloc[0]['phrase']
                return pd.Series({
                    'tfidf_sum': total_tfidf,
                    'freq': total_freq,
                    'phrase': best_phrase
                })

            # 優先級 2: 否則，套用 v10 的邏輯 (處理單位詞)
            CRITICAL_TERMS = {'without', 'change', 'wash', 'dirty',
                              'same', 'no', 'not', 'expensive', 'overpriced', 'too'}
            UNDESIRABLE_ENDINGS = {'time', 'minute', 'hour',
                                   'day', 'week', 'month', 'year', 'second'}

            semantic_phrases = candidates[candidates['phrase'].apply(
                lambda x: any(t in x.split('_') for t in CRITICAL_TERMS))]

            if not semantic_phrases.empty:
                best_phrase = semantic_phrases.iloc[0]['phrase']
            else:
                desirable_phrases = candidates[
                    (~candidates['phrase'].apply(
                        lambda x: x.split('_')[-1] in UNDESIRABLE_ENDINGS))
                ].sort_values(by=['freq', 'tfidf_sum'], ascending=False)

                if not desirable_phrases.empty:
                    best_phrase = desirable_phrases.iloc[0]['phrase']
                else:
                    best_phrase = candidates.iloc[0]['phrase']

            return pd.Series({
                'tfidf_sum': total_tfidf,
                'freq': total_freq,
                'phrase': best_phrase
            })

        try:
            aggregated = df.groupby(
                ['aspect', 'sentiment', 'head_noun']).apply(aggregate_group)
        except TypeError:
            aggregated = df.groupby(
                ['aspect', 'sentiment', 'head_noun']).apply(aggregate_group)

        aggregated = aggregated.reset_index()

        filtered_df = aggregated[aggregated['freq'] > 1]
        if filtered_df.empty and not aggregated.empty:
            filtered_df = aggregated

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
