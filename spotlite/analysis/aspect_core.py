"""
Core functions for aspect-based analysis:
- cleaning review text
- extracting meaningful phrases
- aspect assignment (embedding + seeds)
- sentiment classification (phrase-level & rating-based)
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import logging
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util

from spotlite.analysis.utils_text import clean_en

logger = logging.getLogger(__name__)

# Load spaCy small model once (performance boost)
try:
    NLP = spacy.load("en_core_web_sm")
except Exception as exc:
    logger.warning(
        "spaCy model en_core_web_sm not found. Install via: python -m spacy download en_core_web_sm"
    )
    raise


# Sentence-transformers model for aspect embedding similarity
try:
    ASPECT_EMB_MODEL = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
except Exception as exc:
    logger.error("Failed to load aspect embedding model: %s", exc)
    raise


# -----------------------------------------------------------------------------
# 1. Clean review text
# -----------------------------------------------------------------------------


def clean_review_text(text: str, protected_map: Dict[str, str]) -> str:
    """
    Clean English text and apply phrase protection using the
    given protected_map (由 DomainConfig 提供).
    """
    if not text:
        return ""

    s = text.lower()
    protected_phrases = protected_map.get("protected_phrases")
    if isinstance(protected_phrases, dict):
        items = protected_phrases.items()
    else:
        # Back-compat: protected_map itself is the phrase->merged mapping
        items = protected_map.items()

    for phrase, merged in items:
        if phrase and merged:
            s = s.replace(str(phrase), str(merged))

    s = clean_en(s)
    return s


# -----------------------------------------------------------------------------
# 2. Extract adjective–noun phrases (core feature extraction)
# -----------------------------------------------------------------------------


def extract_adj_noun_phrases(
    text: str,
    max_ngram: int = 4,
    use_lemma: bool = True,
) -> List[str]:
    """
    ULTRA-AGGRESSIVE noun phrase extractor.

    目標：最大化抓出食物 / 飲料 / 嗜好品等名詞片語。

    抓法包含：
    - ADJ + (NOUN/PROPN)+ 例如: amazing matcha, fresh seafood
    - (NOUN/PROPN)+ chain: blueberry matcha, matcha latte, oat milk latte
    - 多元素複合 chain: soft serve ice cream
    - noun_chunks 加強版
    - fallback: 單字名詞
    """
    if not text:
        return []

    doc = NLP(text)

    def norm_token(t: spacy.tokens.Token) -> str:
        """
        Normalize token text.
        - ADJ: surface form (for sentiment seeds)
        - NOUN / PROPN: lemma if enabled, else surface
        """
        if t.pos_ in {"NOUN", "PROPN"} and use_lemma:
            return t.lemma_.lower()
        return t.text.lower()

    phrases: set[str] = set()

    # -------------------------------------------------------------
    # 1) 針對每個 NOUN/PROPN，向左抓所有 ADJ/NOUN/PROPN（更 aggressive）
    # -------------------------------------------------------------
    for token in doc:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue

        left_mods: List[spacy.tokens.Token] = []
        for child in token.lefts:
            if child.pos_ in {"ADJ", "NOUN", "PROPN"} and child.dep_ in {
                "amod",
                "compound",
                "nmod",
                "attr",
                "poss",
            }:
                left_mods.append(child)

        left_mods = sorted(left_mods, key=lambda t: t.i)
        full_chain = left_mods + [token]

        if 1 < len(full_chain) <= max_ngram:
            raw_phrase = "_".join(norm_token(t)
                                  for t in full_chain if t.is_alpha)
            if raw_phrase:
                canon = canonicalize_phrase(raw_phrase)
                phrases.add(canon or raw_phrase)

    # -------------------------------------------------------------
    # 2) noun_chunks 推到最大力度（只用 alpha token）
    # -------------------------------------------------------------
    for chunk in doc.noun_chunks:
        # drop determiners / stopwords like "the", "a" so we don't create phrases like "the_food"
        tokens = [
            norm_token(t)
            for t in chunk
            if t.is_alpha and not t.is_stop and t.pos_ != "DET"
        ]
        if 1 < len(tokens) <= max_ngram:
            raw_phrase = "_".join(tokens)
            if raw_phrase:
                canon = canonicalize_phrase(raw_phrase)
                phrases.add(canon or raw_phrase)

    # -------------------------------------------------------------
    # 3) 額外：依序掃 token 序列，直接組合 multi-NOUN sequence（更 aggressive）
    #    例如：
    #    oat milk latte → oat_milk_latte
    #    soft serve ice cream → soft_serve_ice_cream
    # -------------------------------------------------------------
    # only allow content tokens to start/participate in sequences (avoid "the_food")
    linear_tokens = [
        t for t in doc
        if t.is_alpha and not t.is_stop and t.pos_ in {"NOUN", "PROPN", "ADJ"}
    ]
    n = len(linear_tokens)

    for i in range(n):
        seq: List[spacy.tokens.Token] = [linear_tokens[i]]

        for j in range(i + 1, n):
            if linear_tokens[j].pos_ in {"NOUN", "PROPN", "ADJ"}:
                seq.append(linear_tokens[j])
            else:
                break

            if 1 < len(seq) <= max_ngram:
                raw_phrase = "_".join(norm_token(t) for t in seq)
                if raw_phrase:
                    canon = canonicalize_phrase(raw_phrase)
                    phrases.add(canon or raw_phrase)

    # -------------------------------------------------------------
    # 4) fallback: 單一名詞
    # -------------------------------------------------------------
    for token in doc:
        if token.pos_ in {"NOUN", "PROPN"} and token.text.isalpha():
            raw_phrase = norm_token(token)
            canon = canonicalize_phrase(raw_phrase)
            phrases.add(canon or raw_phrase)

    return list(phrases)


# -----------------------------------------------------------------------------
# 3. Aspect assignment
# -----------------------------------------------------------------------------


def canonicalize_phrase(phrase: str) -> Optional[str]:
    """
    Normalize an evaluative phrase into canonical form: adj_noun.

    Examples:
        sandwich_is_expensive      -> expensive_sandwich
        very_expensive_sandwich    -> expensive_sandwich
        sandwich_was_too_expensive -> expensive_sandwich
    """
    if not phrase:
        return None

    text = phrase.replace("_", " ").strip()
    if not text:
        return None

    doc = NLP(text)
    pairs: List[Tuple[str, str]] = []

    for token in doc:
        # Case 1: adjectival modifier (e.g., expensive sandwich)
        if token.pos_ == "ADJ" and token.dep_ == "amod":
            head = token.head
            if head.pos_ in {"NOUN", "PROPN"}:
                pairs.append((token.text.lower(), head.lemma_.lower()))

        # Case 2: adjectival complement (e.g., sandwich is expensive)
        if token.pos_ == "ADJ" and token.dep_ in {"acomp", "attr"}:
            # Look for nominal subject linked to this adjective
            for ancestor in token.ancestors:
                for child in ancestor.children:
                    if child.dep_ == "nsubj" and child.pos_ in {"NOUN", "PROPN"}:
                        pairs.append(
                            (token.text.lower(), child.lemma_.lower()))

    if not pairs:
        return None

    # Choose the shortest (most canonical) pair
    adj, noun = sorted(pairs, key=lambda x: len(x[0]) + len(x[1]))[0]
    return f"{adj}_{noun}"


def assign_aspect(
    phrase: str,
    model: SentenceTransformer,
    aspect_seed_embs: Dict[str, Tuple[List[str], "np.ndarray"]],
    sim_threshold: float = 0.25,
) -> Optional[str]:
    """Assign an aspect to a phrase using embedding similarity.

    `aspect_seed_embs` must be a mapping:
        aspect -> (seed_words, seed_embeddings)
    where seed_embeddings are pre-computed once per aspect.
    """
    p = phrase.lower().replace("_", " ").strip()

    # Defensive guard: callers sometimes accidentally pass the seed-embedding dict
    # into the `model` parameter (e.g., via positional args). Fail fast with a
    # clear message instead of `AttributeError: 'dict' object has no attribute 'encode'`.
    if not hasattr(model, "encode"):
        raise TypeError(
            "assign_aspect(): `model` must be a SentenceTransformer (has .encode()). "
            f"Got {type(model).__name__}. Did you accidentally pass `aspect_seed_embs` as the second argument?"
        )

    if not p:
        return None

    ph_emb = model.encode(
        [p],
        convert_to_tensor=True,
        show_progress_bar=True,
    )

    best_aspect: Optional[str] = None
    best_sim = -1.0

    for aspect, (_words, seed_emb) in aspect_seed_embs.items():
        sim = util.cos_sim(ph_emb, seed_emb).max().item()
        if sim > best_sim:
            best_sim = sim
            best_aspect = aspect

    if best_aspect is None or best_sim < sim_threshold:
        return None
    return best_aspect


def phrase_sentiment(
    phrase: str,
    seeds_pos: List[str],
    seeds_neg: List[str],
) -> Optional[str]:
    """
    Determine sentiment of a phrase based on positive/negative seed lists.
    """
    if not phrase:
        return None

    p = phrase.lower()

    # positive hit
    for s in seeds_pos:
        if s in p:
            return "pos"

    # negative hit
    for s in seeds_neg:
        if s in p:
            return "neg"

    return None


# -----------------------------------------------------------------------------
# 4. Sentiment classification
# -----------------------------------------------------------------------------


def rating_to_sentiment(rating: Optional[float]) -> str:
    """
    Convert numeric rating (1–5) to sentiment label.
    """
    if rating is None:
        return "neutral"

    if rating >= 4:
        return "pos"
    if rating <= 2:
        return "neg"
    return "neutral"
