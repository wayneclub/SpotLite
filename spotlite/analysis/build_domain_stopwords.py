#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# 從你現有 project 拿這兩個（已經寫好的）
from spotlite.analysis.keywords import clean_en, load_reviews_json

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

# ---- 預設參數（不再從 configs.json 讀）----
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MIN_TOKEN_LEN = 3
DEFAULT_MIN_DOC_FREQ = 3
DEFAULT_MAX_VOCAB_SIZE = 5000
DEFAULT_ASPECT_SIM_THRESHOLD = 0.25


# ---------- 小工具 ----------

def build_doc_freq(docs: List[str]) -> Counter:
    df_counter = Counter()
    for doc in docs:
        tokens = doc.split()
        df_counter.update(set(tokens))
    return df_counter


def filter_candidate_vocab(
    df_counter: Counter,
    min_doc_freq: int,
    min_token_len: int,
    max_vocab_size: int
) -> List[str]:
    items = [
        (token, df)
        for token, df in df_counter.items()
        if len(token) >= min_token_len and df >= min_doc_freq
    ]
    items.sort(key=lambda x: x[1], reverse=True)
    if max_vocab_size > 0:
        items = items[:max_vocab_size]
    return [t for t, _ in items]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def load_aspect_seeds(path: Path) -> Dict[str, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def compute_aspect_centroids(
    model: SentenceTransformer,
    aspect_cfg: Dict[str, Dict[str, List[str]]]
) -> Dict[str, np.ndarray]:
    centroids = {}
    for aspect, cfg in aspect_cfg.items():
        seeds = cfg.get("seeds_pos", []) + cfg.get("seeds_neg", [])
        seeds = [s for s in seeds if s.strip()]
        if not seeds:
            logger.warning("Aspect '%s' has no seeds, skip.", aspect)
            continue
        embs = model.encode(seeds, convert_to_numpy=True)
        centroids[aspect] = embs.mean(axis=0)
        logger.info("Aspect '%s' centroid from %d seeds.", aspect, len(seeds))
    return centroids


def assign_token_to_aspect(
    token_emb: np.ndarray,
    centroids: Dict[str, np.ndarray]
) -> Tuple[str, float]:
    if not centroids:
        return "", 0.0
    best_aspect = ""
    best_sim = -1.0
    for aspect, c in centroids.items():
        sim = cosine_similarity(token_emb, c)
        if sim > best_sim:
            best_sim = sim
            best_aspect = aspect
    return best_aspect, best_sim


# ---------- 主流程 ----------

def build_domain_stopwords(
    input_path: Path,
    output_path: Path,
    aspect_seeds_path: Path,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    min_token_len: int = DEFAULT_MIN_TOKEN_LEN,
    min_doc_freq: int = DEFAULT_MIN_DOC_FREQ,
    max_vocab_size: int = DEFAULT_MAX_VOCAB_SIZE,
    aspect_sim_threshold: float = DEFAULT_ASPECT_SIM_THRESHOLD,
    base_stopwords_json: Path | None = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    aspect_seeds_path = Path(aspect_seeds_path)

    logger.info("Loading reviews from %s", input_path)
    raw_texts = load_reviews_json(input_path)
    docs = [clean_en(t) for t in raw_texts if t and t.strip()]
    logger.info("Total reviews: %d, after cleaning: %d",
                len(raw_texts), len(docs))

    df_counter = build_doc_freq(docs)
    logger.info("Vocabulary size: %d", len(df_counter))

    vocab = filter_candidate_vocab(
        df_counter,
        min_doc_freq=min_doc_freq,
        min_token_len=min_token_len,
        max_vocab_size=max_vocab_size,
    )
    logger.info("Candidate vocab size after filtering: %d", len(vocab))
    if not vocab:
        logger.warning("No candidate tokens. Abort.")
        return

    logger.info("Loading aspect seeds from %s", aspect_seeds_path)
    aspect_cfg = load_aspect_seeds(aspect_seeds_path)

    logger.info("Loading embedding model: %s", embedding_model_name)
    model = SentenceTransformer(embedding_model_name)

    centroids = compute_aspect_centroids(model, aspect_cfg)

    logger.info("Encoding %d tokens...", len(vocab))
    token_embs = model.encode(vocab, convert_to_numpy=True, batch_size=128)

    stopword_candidates = []
    token_info = []
    for token, emb in zip(vocab, token_embs):
        best_aspect, best_sim = assign_token_to_aspect(emb, centroids)
        df = df_counter[token]
        is_outlier = best_sim < aspect_sim_threshold
        if is_outlier:
            stopword_candidates.append(token)
        token_info.append({
            "token": token,
            "df": int(df),
            "best_aspect": best_aspect,
            "best_sim": float(best_sim),
            "is_outlier": bool(is_outlier),
        })

    logger.info(
        "Outlier tokens (domain stopword candidates): %d / %d",
        len(stopword_candidates),
        len(vocab),
    )

    existing = set()
    if base_stopwords_json is not None and base_stopwords_json.exists():
        with open(base_stopwords_json, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, dict) and "stopwords" in data:
                    existing.update(data["stopwords"])
                elif isinstance(data, list):
                    existing.update(data)
            except json.JSONDecodeError:
                logger.warning(
                    "Failed to parse %s, ignore base stopwords.", base_stopwords_json)

    final_stopwords = sorted(existing.union(stopword_candidates))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_obj = {
        "stopwords": final_stopwords,
        "meta": {
            "source": str(input_path),
            "num_reviews": len(raw_texts),
            "num_docs_cleaned": len(docs),
            "min_token_len": min_token_len,
            "min_doc_freq": min_doc_freq,
            "max_vocab_size": max_vocab_size,
            "aspect_sim_threshold": aspect_sim_threshold,
            "embedding_model": embedding_model_name,
            "aspect_seeds_path": str(aspect_seeds_path),
        },
        "token_info_sample": token_info[:200]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    logger.info("Saved domain stopwords to %s (total %d)",
                output_path, len(final_stopwords))


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build domain-specific stopwords for restaurant reviews."
    )
    p.add_argument("--input", required=True,
                   help="reviews JSON, e.g. Holbox_reviews.json")
    p.add_argument("--output", required=True,
                   help="output JSON, e.g. config/domain_stopwords.json")
    p.add_argument("--aspect-seeds-json", required=True,
                   help="aspect_seeds.json 路徑")
    p.add_argument("--base-stopwords-json", default=None,
                   help="(選用) 舊的 domain_stopwords.json")
    p.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    p.add_argument("--min-token-len", type=int, default=DEFAULT_MIN_TOKEN_LEN)
    p.add_argument("--min-doc-freq", type=int, default=DEFAULT_MIN_DOC_FREQ)
    p.add_argument("--max-vocab-size", type=int,
                   default=DEFAULT_MAX_VOCAB_SIZE)
    p.add_argument("--aspect-sim-threshold", type=float,
                   default=DEFAULT_ASPECT_SIM_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()
    build_domain_stopwords(
        input_path=Path(args.input),
        output_path=Path(args.output),
        aspect_seeds_path=Path(args.aspect_seeds_json),
        embedding_model_name=args.embedding_model,
        min_token_len=args.min_token_len,
        min_doc_freq=args.min_doc_freq,
        max_vocab_size=args.max_vocab_size,
        aspect_sim_threshold=args.aspect_sim_threshold,
        base_stopwords_json=Path(
            args.base_stopwords_json) if args.base_stopwords_json else None,
    )


if __name__ == "__main__":
    main()
