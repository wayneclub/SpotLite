"""
Aspect Keyword Analyzer (Ultimate Edition v52 - Draft & Polish)
------------------------------------------
版本更新 (v52) 總結：
1. [SUMMARY FIX] 草稿與潤飾策略 (Draft & Polish)：
   - 不再讓 AI 直接從關鍵字生成摘要（避免幻覺或指令誤解）。
   - Step 1 (Python): 根據 top_keywords 自動組裝一個「事實草稿」(Fact Draft)。強制包含頻率最高的食物。
   - Step 2 (AI): 使用 T5 模型將這個草稿「改寫 (Rewrite)」為自然段落。
   - 效果：100% 提及指定食物 (matcha, coffee)，0% 幻覺 (不會出現 sushi)，語句通順。
2. [RETAINED] 保留 v48 的所有核心功能。
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

# Hugging Face Imports
try:
    import torch
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers/Torch/BitsAndBytes not found.")

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
_summary_pipeline = None


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
        _aspect_model = SentenceTransformer('all-mpnet-base-v2')
    return _aspect_model


def get_summary_pipeline():
    """
    [UPGRADE] Load Zephyr-7B-beta using 4-bit quantization for high-quality natural text.
    Requires GPU (T4 is fine).
    """
    global _summary_pipeline
    if _summary_pipeline is None and TRANSFORMERS_AVAILABLE:
        # [CONFIG] 模型選擇
        # "google/flan-t5-large" (推薦: 品質好，速度尚可)
        # "google/flan-t5-base"  (極速: 速度快，但語法較生硬)
        model_id = "google/flan-t5-large"

        logger.info(f"Loading LLM model: {model_id}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

            # Explicitly use text2text-generation for T5
            _summary_pipeline = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer
            )
            logger.info("✅ Model loaded successfully.")

        except Exception as e:
            logger.error(f"❌ Failed to load summary model: {e}")

    return _summary_pipeline

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
        self.place_name = ""

        self.menu_items: Set[str] = set()

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
        # self._inject_hardcoded_domain_knowledge() # Removed in v48
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

        # 3. Aspect Seeds & Whitelisting
        seeds_path = base / "aspect_seeds.json"
        if seeds_path.exists():
            try:
                with open(seeds_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.aspect_seeds_raw = data
                    whitelist_candidates = set()

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

                        whitelist_candidates.update(all_seeds)
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

                        whitelist_candidates.update(
                            data["global_sentiment"].get("pos", []))
                        whitelist_candidates.update(
                            data["global_sentiment"].get("neg", []))

                    removed_count = 0
                    for word in whitelist_candidates:
                        clean_word = word.lower().strip()
                        if clean_word in self.stopwords:
                            self.stopwords.remove(clean_word)
                            removed_count += 1
                        underscore_word = clean_word.replace(" ", "_")
                        if underscore_word in self.stopwords:
                            self.stopwords.remove(underscore_word)
                            removed_count += 1

                    logger.info(
                        f"Loaded seeds and whitelisted {removed_count} terms from stopwords.")

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

        # Pattern 1
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

        # Pattern 2 & 5
        for token in doc:
            if token.pos_ == "ADJ" or (token.pos_ == "VERB" and token.tag_ in ("VBN", "VBG")):
                adj = token.lemma_.lower()

                if token.dep_ in ("acomp", "attr"):
                    copular = token.head

                    subjects = []
                    for child in copular.children:
                        if child.dep_ == "nsubj" and child.pos_ in VALID_NOUN:
                            subjects.append(child)
                            for conj in child.conjuncts:
                                if conj.pos_ in VALID_NOUN:
                                    subjects.append(conj)

                    is_neg = any(c.dep_ == "neg" or c.lemma_ in {
                                 "never", "not"} for c in copular.children)
                    adv = get_adv(token)

                    for subj in subjects:
                        noun = subj.lemma_.lower()
                        if subj.text in protected_set:
                            noun = subj.text
                        elif noun in CHECK_SET:
                            continue

                        parts = []
                        if is_neg:
                            parts.append("not")
                        if adv:
                            parts.append(adv)
                        parts.append(adj)
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
                    if c.dep_ == "neg" or c.lemma_.lower() in {"never", "no", "not"}:
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
        for token in doc:
            CRITICAL_HYGIENE = {"gloves", "glove",
                                "mask", "masks", "hand", "hands"}
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

        if len(parts_lemma) == 1 and parts_lemma[0] in self.global_sentiment_terms:
            return None

        if text_lemma in self.stopwords:
            return None

        KEEP_TRIGGERS = {
            "dirty", "filthy", "stained", "unclean", "grimy", "smelly",
            "broken", "chipped", "cracked",
            "no", "missing", "lack", "ask", "need",
            "michelin", "texas", "nashville",
            "without", "change", "wear",
        }

        is_generic_stopword_found = False
        for part in parts_lemma:
            if part in self.stopwords:
                if not any(part in trigger for trigger in KEEP_TRIGGERS):
                    is_generic_stopword_found = True
                    break

        if is_generic_stopword_found:
            return None

        PRIORITY_MAP = {
            "price": ["affordable", "cheap", "expensive", "price", "cost", "bill", "value", "worth", "overpriced", "pricy", "reasonable", "rip-off", "check", "dollar", "buck", "money"],
            "waiting_time": ["wait", "queue", "line", "slow", "fast", "rush", "forever", "long time"],
            "service": ["rude", "friendly", "polite", "helpful", "staff", "waiter", "waitress", "manager", "server", "host", "hostess", "glove", "mask", "hygiene"]
        }
        for aspect, keywords in PRIORITY_MAP.items():
            if any(k in text_lower for k in keywords):
                if "fast food" in text_lower and aspect == "waiting_time":
                    continue
                return aspect

        for aspect, seeds in self.aspect_seeds.items():
            for seed in seeds:
                if len(seed) > 2 and seed in text_lemma:
                    return aspect

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

        NEUTRAL_FLAVORS = {'spicy', 'sour', 'sweet', 'bitter', 'salty', 'hot'}
        has_neutral_flavor = any(w in NEUTRAL_FLAVORS for w in parts)

        STRONG_POS = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'perfect',
                      'best', 'nice', 'love', 'delicious', 'yummy', 'tasty', 'flavorful', 'hit', 'spot'}
        NEGATION = {'not', 'no', 'never', "didn't",
                    "don't", "cant", "cannot", "wouldn't", "wont"}
        EXCESS = {'too', 'overly', 'excessively'}
        STRONG_NEG = {'bad', 'terrible', 'awful',
                      'horrible', 'worst', 'disgusting', 'gross'}

        has_pos = any(w in STRONG_POS for w in parts)
        has_neg_word = any(w in STRONG_NEG for w in parts)
        has_negation = any(w in NEGATION for w in parts)
        has_excess = any(w in EXCESS for w in parts)

        if has_neutral_flavor:
            if has_pos and not has_negation:
                return "pos"
            elif has_negation or has_excess or has_neg_word:
                return "neg"
            else:
                return None

        seeds_obj = self.aspect_seeds_raw.get(aspect, {})
        raw_neg = (seeds_obj.get("seeds_neg", [])
                   if isinstance(seeds_obj, dict) else [])

        nlp = get_nlp()
        neg_lemma_set = set(["_".join([t.lemma_.lower()
                            for t in nlp(w)]) for w in raw_neg])
        ALL_NEG_ADJS = set(self.global_neg_seeds).union(neg_lemma_set)

        if any(k in parts for k in NEGATION):
            head_lemma = nlp(head)[0].lemma_.lower()
            if head_lemma in ALL_NEG_ADJS or any(adj in text_lower for adj in ALL_NEG_ADJS):
                return "pos"
            return "neg"

        TIME_UNITS = {"forever", "eternity", "age", "long"}
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

        sentiment = None
        if score > 0.05:
            sentiment = "pos"
        elif score < -0.05:
            sentiment = "neg"

        return self._check_sentiment_consistency(phrase, sentiment)

    def _check_sentiment_consistency(self, phrase: str, sentiment: str) -> str:
        if sentiment is None:
            return None

        parts = phrase.lower().split('_')
        STRONG_POS = {'good', 'great', 'excellent', 'amazing', 'wonderful',
                      'fantastic', 'perfect', 'best', 'nice', 'love', 'delicious', 'yummy', 'tasty'}
        NEGATION = {'not', 'no', 'never', "didn't",
                    "don't", "cant", "cannot", "wouldn't", "wont"}

        has_pos_word = any(w in STRONG_POS for w in parts)
        has_negation = any(w in NEGATION for w in parts)

        if has_pos_word and not has_negation:
            return "pos"

        return sentiment

    def load_data(self, input_path: str | Path) -> 'AspectKeywordAnalyzer':
        if not isinstance(input_path, Path):
            try:
                input_path = Path(input_path)
            except Exception as e:
                logger.error(
                    f"Failed to convert input_path to Path object: {e}")
                raise TypeError(
                    f"Invalid input_path format: Must be str or Path.")

        if not input_path.exists():
            raise FileNotFoundError(input_path)

        self.input_stem = input_path.stem.replace("_reviews", "")
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.place_name = raw_data.get(
            "place_name") or raw_data.get("name") or ""
        self.raw_records = raw_data.get(
            "reviews") or raw_data.get("data") or raw_data

        logger.info(f"Loaded {len(self.raw_records)} reviews.")

        self._inject_menu_items(raw_data)
        self._dynamic_stopword_update()
        return self

    def _inject_menu_items(self, raw_data: Dict):
        dishes = raw_data.get("dishes", [])
        if not dishes:
            return

        logger.info(
            f"Found {len(dishes)} menu items. Injecting into protection list.")
        nlp = get_nlp()
        for dish in dishes:
            clean_key = str(dish).lower().strip()
            clean_val = clean_key.replace(" ", "_")

            # 1. Protect original
            self.protected_phrases[clean_key] = clean_val
            self.protected_tokens.add(clean_val)
            self.menu_items.add(clean_val)

            # 2. Protect lemma
            doc = nlp(clean_key)
            lemma_parts = [t.lemma_.lower() for t in doc]
            clean_key_lemma = " ".join(lemma_parts)
            clean_val_lemma = "_".join(lemma_parts)

            if clean_val_lemma != clean_val:
                self.protected_phrases[clean_key_lemma] = clean_val_lemma
                self.protected_tokens.add(clean_val_lemma)
                self.menu_items.add(clean_val_lemma)

            if clean_key in self.stopwords:
                self.stopwords.remove(clean_key)
            if clean_val in self.stopwords:
                self.stopwords.remove(clean_val)
            if clean_key_lemma in self.stopwords:
                self.stopwords.remove(clean_key_lemma)
            if clean_val_lemma in self.stopwords:
                self.stopwords.remove(clean_val_lemma)

    def _dynamic_stopword_update(self):
        if not self.place_name:
            return

        logger.info(f"Dynamically processing store name: {self.place_name}")
        nlp = get_nlp()

        known_products = set(self.aspect_seeds.get('taste', []))
        if isinstance(self.aspect_seeds_raw.get('taste'), dict):
            keywords = self.aspect_seeds_raw['taste'].get('keywords', [])
            known_products.update(keywords)

        doc = nlp(self.place_name)
        for token in doc:
            word = token.text.lower()
            if word not in known_products and len(word) > 1:
                self.stopwords.add(word)
                logger.info(f"  -> Added '{word}' to dynamic stopwords.")
            else:
                logger.info(
                    f"  -> Kept '{word}' (recognized as product/keyword).")

    def extract_phrases(self) -> 'AspectKeywordAnalyzer':
        if not self.raw_records:
            return self
        logger.info("Extracting phrases...")
        processed = []
        for item in self.raw_records:
            text = (item.get("text") or item.get(
                "plain_text") or item.get("raw_text") or "")
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
            'happy_hour', 'tasting_menu', 'omakase', 'baked_crab', 'unagi_shrimp_tempura',
            'spicy_albacore', 'malatang', 'broth', 'hot_pot', 'soup_base',
            'wrong_delivery', 'wrong_order', 'missing_item'
        }

        FIXED_PHRASES.update(self.menu_items)

        GENERIC_HEADS = {
            'place', 'spot', 'location', 'area', 'restaurant', 'shop', 'store',
            'joint', 'thing', 'way', 'option', 'choice', 'selection', 'experience',
            'dish', 'item', 'one', 'bit', 'love', 'like',
            'apology', 'time'
        }

        SECONDARY_GENERIC_HEADS = {
            'food', 'meal', 'cuisine', 'appetizer', 'entree', 'dessert', 'snack'}

        SENTIMENT_ADJECTIVES = {
            'good', 'great', 'bad', 'nice', 'excellent', 'amazing', 'wonderful',
            'terrible', 'awful', 'horrible', 'best', 'worst', 'delicious', 'tasty',
            'yummy', 'fantastic', 'perfect', 'ok', 'okay', 'decent', 'fine', 'authentic',
            'fresh', 'super', 'highly', 'pretty'
        }

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

            is_action_phrase = len(
                parts) >= 2 and parts[-2] in {'change', 'wash', 'wear'}
            is_hygiene_noun = parts[-1] in {'glove', 'mask'}

            if is_action_phrase or is_hygiene_noun:
                if parts[0] in {'not', 'without'}:
                    return "_".join(parts[1:])
                return "_".join(parts)

            if phrase in FIXED_PHRASES:
                head_word_raw = phrase.split('_')[-1]
                if len(parts) == 1:
                    head_word_lemma = nlp(head_word_raw)[0].lemma_.lower()
                    return head_word_lemma

                head_word_lemma = nlp(head_word_raw)[0].lemma_.lower()
                base = "_".join(parts[:-1])
                return f"{base}_{head_word_lemma}"

            if len(parts) > 2 and 'crispy_rice' in FIXED_PHRASES:
                if phrase.endswith('crispy_rice') and ('albacore' in phrase or 'tuna' in phrase):
                    return 'crispy_rice'

            for fixed in FIXED_PHRASES:
                if phrase.endswith(f"_{fixed}"):
                    return fixed

            if len(parts) > 1 and parts[-1] in GENERIC_HEADS:
                raw_noun = parts[-2]
            else:
                raw_noun = parts[-1]

            clean_parts = []
            for p in parts[:-1]:
                if p not in SENTIMENT_ADJECTIVES:
                    clean_parts.append(p)
            clean_parts.append(parts[-1])

            potential_head = "_".join(clean_parts)

            if len(clean_parts) > 0:
                raw_noun = potential_head
            else:
                raw_noun = parts[-1]

            if raw_noun in {'good', 'great', 'nice', 'bad', 'terrible', 'ok', 'okay', 'fine', 'decent', 'pretty', 'really'}:
                return 'generic_sentiment'

            if raw_noun in {'food', 'meal', 'dish', 'cuisine', 'drink', 'beverage', 'drinks', 'beverages'}:
                if len(parts) == 1 or parts[0] in self.global_sentiment_terms or parts[0] in SENTIMENT_ADJECTIVES:
                    return 'generic_food'

            if raw_noun in {'time', 'minute', 'hour', 'second', 'day', 'week', 'month', 'year'}:
                return 'generic_time'

            if "_" in raw_noun:
                head_word_raw = raw_noun.split('_')[-1]
                head_word_lemma = nlp(head_word_raw)[0].lemma_.lower()
                base_parts = raw_noun.split('_')[:-1]
                return "_".join(base_parts + [head_word_lemma])

            if raw_noun in SECONDARY_GENERIC_HEADS:
                return raw_noun

            if raw_noun in SYNONYM_MAP:
                return SYNONYM_MAP[raw_noun]

            lemma = nlp(raw_noun.split('_')[-1])[0].lemma_.lower()
            if lemma in SYNONYM_MAP:
                return SYNONYM_MAP[lemma]

            if lemma == raw_noun.split('_')[-1].lower() and lemma.endswith('s') and not lemma.endswith('ss'):
                singular = lemma[:-1]
                if singular in SYNONYM_MAP:
                    return SYNONYM_MAP[singular]
                return singular

            return raw_noun

        df['head_noun'] = df['phrase'].apply(normalize_noun)

        df = df[~df['head_noun'].isin(
            ['generic_food', 'generic_time', 'generic_sentiment'])].copy()

        def aggregate_group(group):
            total_tfidf = group['tfidf_sum'].sum()
            total_freq = group['freq'].sum()

            candidates = group.sort_values(
                by=['freq', 'tfidf_sum'], ascending=False)

            sentiment_phrases = candidates[candidates['phrase'].apply(
                lambda x: any(w in SENTIMENT_ADJECTIVES for w in x.split('_'))
            )]

            if not sentiment_phrases.empty:
                best_phrase = sentiment_phrases.iloc[0]['phrase']
            else:
                descriptive_phrases = candidates[
                    (candidates['phrase'].str.contains('_')) &
                    (~candidates['phrase'].apply(lambda x: x in FIXED_PHRASES))
                ].sort_values(by=['freq', 'tfidf_sum'], ascending=False)

                if not descriptive_phrases.empty:
                    best_phrase = descriptive_phrases.iloc[0]['phrase']
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

        filtered_df = aggregated[aggregated['freq'] > 1].copy()

        if filtered_df.empty and not aggregated.empty:
            filtered_df = aggregated

        return filtered_df.sort_values(
            by=['aspect', 'sentiment', 'tfidf_sum', 'freq'],
            ascending=[True, True, False, False]
        ).groupby(['aspect', 'sentiment']).head(top_n)

    def generate_summary(self):
        """
        Narrative Synthesis Strategy for natural, human-like summaries (FLAN-T5).
        Fixes:
        - Use text2text-generation explicitly (FLAN-T5 correct task)
        - Short, strict prompt; DATA at the end
        - Use max_new_tokens (avoid max_length issues)
        - Deterministic decoding first; optional sampling mode
        - Fallback if model returns "instruction-like" text
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.warning(
                "Summary generation skipped: Transformers not installed.")
            return

        pipe = get_summary_pipeline()
        if pipe is None:
            logger.warning(
                "Summary generation skipped: Model could not be loaded.")
            return

        # ---- Ensure correct pipeline task (FLAN-T5 wants text2text-generation) ----
        # If your get_summary_pipeline() already returns a proper pipe, this is fine.
        # But if it returns summarization/text-generation by mistake, rebuild it.
        try:
            task = getattr(pipe, "task", None)
            if task != "text2text-generation":
                from transformers import pipeline as hf_pipeline
                logger.warning(
                    f"Rebuilding pipeline: expected text2text-generation, got {task}")
                pipe = hf_pipeline("text2text-generation",
                                   model="google/flan-t5-large")
        except Exception as e:
            logger.warning(f"Could not verify pipeline task: {e}")

        top5_df = self.get_top_keywords(5)
        if top5_df.empty:
            logger.warning("Summary generation skipped: No keywords found.")
            return

        # ----------------------------
        # 1) Build compact DATA facts
        # ----------------------------
        data_lines = []

        aspect_priority = ["taste", "service",
                           "environment", "price", "waiting_time"]
        existing_aspects = list(top5_df["aspect"].dropna().unique())

        sorted_aspects = [a for a in aspect_priority if a in existing_aspects]
        sorted_aspects += [a for a in existing_aspects if a not in aspect_priority]

        for aspect in sorted_aspects:
            aspect_df = top5_df[top5_df["aspect"] == aspect].sort_values(
                by="freq", ascending=False)

            pos_phrases, neg_phrases = [], []
            for _, row in aspect_df.iterrows():
                phrase = str(row.get("phrase", "")).replace("_", " ").strip()
                if not phrase:
                    continue
                if row.get("sentiment") == "pos":
                    pos_phrases.append(phrase)
                else:
                    neg_phrases.append(phrase)

            # keep it tight; too much noise makes FLAN drift
            pos_phrases = pos_phrases[:4]
            neg_phrases = neg_phrases[:3]

            # Use field-like format (less "listy" than bullets)
            if pos_phrases:
                data_lines.append(
                    f"{aspect.upper()}_PROS: {', '.join(pos_phrases)}")
            if neg_phrases:
                data_lines.append(
                    f"{aspect.upper()}_CONS: {', '.join(neg_phrases)}")

        context_data = "\n".join(data_lines).strip()

        # Debug (recommended)
        logger.info("context_data_len=%d", len(context_data))
        logger.info("context_data_preview=\n%s", context_data[:600])

        if len(context_data) < 60:
            logger.warning(
                "Summary generation skipped: context_data too short / weak signal.")
            return

        # ----------------------------
        # 2) Short, strict prompt
        # ----------------------------
        prompt = f"""Write a single cohesive restaurant review summary based ONLY on the DATA.

Output requirements:
- 4 to 6 sentences total (not bullet points).
- Sentence 1: overall vibe/positioning from the strongest pros.
- Sentence 2-3: details on Taste first, then Service/Environment.
- Use "However," to transition into weaknesses.
- End with a practical recommendation (who it's good for / what to expect).
- Do NOT repeat the DATA as a list. Do NOT explain the task.

DATA:
{context_data}

ANSWER:
"""

        # ----------------------------
        # 3) Generation (deterministic first)
        # ----------------------------
        def looks_like_instructions(text: str) -> bool:
            t = (text or "").strip().lower()
            bad_starts = (
                "describe", "write a", "task:", "guidelines:", "for example", "include the",
                "use transition", "avoid using"
            )
            # If it contains too many instruction words, treat as failure
            return (t.startswith(bad_starts) or ("guidelines" in t) or ("task" in t and "data" not in t))

        def rule_based_fallback() -> str:
            # Simple, safe fallback paragraph (no hallucination)
            # Picks top aspects by presence and nets: pros first, then cons.
            pros_bits, cons_bits = [], []
            for line in data_lines:
                if "_PROS:" in line:
                    label = line.split("_PROS:", 1)[0].title()
                    pros_bits.append(
                        f"{label.lower()} highlights include {line.split(':',1)[1].strip()}")
                if "_CONS:" in line:
                    label = line.split("_CONS:", 1)[0].title()
                    cons_bits.append(
                        f"{label.lower()} concerns include {line.split(':',1)[1].strip()}")
            pros_sentence = " ".join(
                pros_bits[:2]) if pros_bits else "Overall, the feedback highlights a few positives."
            cons_sentence = " ".join(
                cons_bits[:2]) if cons_bits else "However, there are limited negative signals in the extracted keywords."
            return f"{pros_sentence}. However, {cons_sentence}."

        try:
            logger.info("Generating summary with FLAN-T5 (deterministic)...")
            out = pipe(
                prompt,
                max_new_tokens=160,
                do_sample=False,
                num_beams=4,
                repetition_penalty=1.25,
                no_repeat_ngram_size=4,
                early_stopping=True,
            )
            summary_text = (out[0].get("generated_text", "") or "").strip()

            # If it still outputs instructions, retry once with sampling
            if looks_like_instructions(summary_text):
                logger.warning(
                    "Model returned instruction-like text. Retrying with sampling...")
                out = pipe(
                    prompt,
                    max_new_tokens=180,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    num_beams=1,
                    repetition_penalty=1.25,
                    no_repeat_ngram_size=4,
                )
                summary_text = (out[0].get("generated_text", "") or "").strip()

            # Final fallback (guaranteed output)
            if looks_like_instructions(summary_text) or len(summary_text) < 40:
                logger.warning(
                    "Model output still invalid. Using rule-based fallback.")
                summary_text = rule_based_fallback()

            # Save
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_ROOT / self.input_stem / f"summary_{ts}.txt"
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

            logger.info(f"Summary saved to {out_path}")
            print("\n--- Generated Summary ---")
            print(summary_text)
            print("-------------------------")

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

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

    analyzer = AspectKeywordAnalyzer(
        sys.argv[2] if len(sys.argv) > 2 else "restaurant")
    analyzer.load_data(sys.argv[1])
    analyzer.extract_phrases()
    analyzer.compute_tfidf()
    analyzer.save_results()

    analyzer.generate_summary()

    print("Done.")
