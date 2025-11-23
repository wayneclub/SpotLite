"""
SpotLite Keyword Analyzer (English-only)
- Input: /mnt/data/reviews.json  (list[str] or list[dict] with keys like "text"/"content")
- Output CSVs under /mnt/data/spotlite_outputs:
  - top_keywords.csv
  - keyword_clusters.csv
  - keyword_examples.csv
  - keywords_full.csv
  - SUMMARY.txt
"""

import json
import re
import math
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.cluster import AgglomerativeClustering

from spotlite.config.config import load_config


# ---------- Config via configs.json ----------
# Load separate configs
GENERAL_CFG = load_config("configs.json")
KEYWORDS_CFG = load_config("keywords.json")

# Output path comes from configs.json paths block
PATH_CFG = GENERAL_CFG.get("paths", {})
OUTPUT_ROOT = Path(PATH_CFG.get("keywords_output_root", "outputs"))

# Keyword extraction settings
MAX_FEATURES = KEYWORDS_CFG.get("max_features", 1000)
N_TOP = KEYWORDS_CFG.get("n_top", 50)
# JSON cannot store tuples directly; expect a list like [1, 2]
NGRAM_RANGE = tuple(KEYWORDS_CFG.get("ngram_range", [1, 2]))
MIN_DOC_FREQ = KEYWORDS_CFG.get("min_doc_freq", 2)

# ---------- Logger initialization ----------
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

DOMAIN_STOP = {
    # Restaurant/Hotel generic words to reduce noise
    "restaurant", "restaurants", "food", "place", "places", "menu", "dish", "dishes",
    "hotel", "hotels", "room", "rooms", "stay", "stayed", "night", "nights",
    "staff", "service", "services", "people", "person", "location", "area", "spot",
    "come", "came", "going", "went", "got", "get", "one", "two", "three", "four", "five",
    "really", "very", "quite", "bit", "lot", "little", "awesome", "amazing", "great",
    "good", "bad", "okay", "nice", "love", "loved", "like", "liked", "dislike", "disliked",
    "best", "worst", "experience", "experiences", "review", "reviews", "definitely",
    "highly", "recommend", "recommended", "would", "could", "should", "also"
}
STOPWORDS = list(set(ENGLISH_STOP_WORDS).union(DOMAIN_STOP))

URL_RE = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def load_reviews_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "reviews" in data:
        data = data["reviews"]
    raw = []
    for item in data:
        if isinstance(item, str):
            txt = item
        elif isinstance(item, dict):
            txt = item.get("text") or item.get("reviewText") or item.get(
                "content") or item.get("body") or item.get("review") or ""
        else:
            txt = ""
        if txt and txt.strip():
            raw.append(txt)
    return raw


def clean_en(s: str) -> str:
    s = s.lower()
    s = URL_RE.sub(" ", s)
    s = NON_ALPHA_RE.sub(" ", s)
    s = MULTI_SPACE_RE.sub(" ", s).strip()
    return s


def snippet(text: str, query: str, width: int = 120) -> str:
    text = text.replace("\n", " ")
    m = re.search(re.escape(query), text, flags=re.IGNORECASE)
    if not m:
        return text[:width] + ("..." if len(text) > width else "")
    start = max(0, m.start() - width//2)
    end = min(len(text), m.end() + width//2)
    out = text[start:end]
    if start > 0:
        out = "..." + out
    if end < len(text):
        out = out + "..."
    return out


def analyze_keywords(input_path: Path):
    """
    Run keyword analysis for the given reviews JSON file.

    Parameters
    ----------
    input_path : Path or str
        Path to a reviews JSON file (e.g. '<place_name>_reviews.json').

    Returns
    -------
    Path
        The output directory where result files are written.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found.")

    # Derive place name and per-place output directory
    place_name = input_path.stem
    if place_name.endswith("_reviews"):
        place_name = place_name[:-8]  # strip trailing '_reviews'
    output_dir = OUTPUT_ROOT / place_name
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_texts = load_reviews_json(input_path)
    docs = [clean_en(t) for t in raw_texts if t.strip()]
    if not docs:
        raise ValueError("No valid texts after cleaning.")

    # ---- TF-IDF keywords ----
    vec = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        max_features=MAX_FEATURES,
        min_df=MIN_DOC_FREQ,
        stop_words=STOPWORDS
    )
    X = vec.fit_transform(docs)
    terms = vec.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).ravel()

    order = np.argsort(scores)[::-1]
    top_idx = order[:min(N_TOP, len(order))]
    top_terms = terms[top_idx]
    top_scores = scores[top_idx]
    top_df = pd.DataFrame({
        "rank": np.arange(1, len(top_terms)+1),
        "keyword": top_terms,
        "tfidf_sum": np.round(top_scores, 6)
    })

    # ---- Representative example for each keyword ----
    term_to_col = {t: i for i, t in enumerate(terms)}
    rows = []
    for kw in top_terms:
        j = term_to_col[kw]
        col = X[:, j].toarray().ravel()
        d_idx = int(col.argmax()) if col.max() > 0 else None
        rows.append({
            "keyword": kw,
            "best_doc_index": d_idx,
            "best_tfidf": float(col[d_idx]) if d_idx is not None else 0.0,
            "example_snippet": snippet(raw_texts[d_idx], kw) if d_idx is not None else ""
        })
    examples_df = pd.DataFrame(rows)

    # ---- Simple clustering by co-occurrence PMI-like similarity ----
    doc_bin = (X[:, top_idx] > 0).astype(int)
    n_docs = doc_bin.shape[0]
    if len(top_terms) >= 4 and n_docs >= 2:
        co_mat = (doc_bin.T @ doc_bin).toarray()
        df_term = doc_bin.sum(axis=0).A1 + 1e-9
        p_i = df_term / n_docs
        p_ij = (co_mat / n_docs) + 1e-12
        n_pmi = np.log(p_ij / (p_i.reshape(-1, 1) * p_i.reshape(1, -1)))
        sim = 1 / (1 + np.exp(-n_pmi))  # logistic normalization to [0,1]
        dist = 1 - sim
        np.fill_diagonal(dist, 0.0)
        n_clusters = max(3, min(10, int(round(math.sqrt(len(top_terms))))))
        try:
            clust = AgglomerativeClustering(
                metric="precomputed", linkage="average", n_clusters=n_clusters)
            labels = clust.fit_predict(dist)
        except Exception:
            labels = np.zeros(len(top_terms), dtype=int)
    else:
        labels = np.zeros(len(top_terms), dtype=int)

    clusters_df = pd.DataFrame({"keyword": top_terms, "cluster": labels}).sort_values(
        ["cluster", "keyword"]).reset_index(drop=True)

    # ---- Save all ----
    top_df.to_csv(output_dir / "top_keywords.csv", index=False)
    clusters_df.to_csv(output_dir / "keyword_clusters.csv", index=False)
    examples_df.to_csv(output_dir / "keyword_examples.csv", index=False)

    full_df = top_df.merge(examples_df, on="keyword", how="left").merge(
        clusters_df, on="keyword", how="left")
    full_df.to_csv(output_dir / "keywords_full.csv", index=False)

    with open(output_dir / "SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(f"Total reviews: {len(raw_texts)}\n")
        f.write(f"After cleaning: {len(docs)}\n")
        f.write(f"Vocabulary size: {len(terms)}\n")
        f.write(f"Top keywords extracted: {len(top_terms)}\n")
        f.write(
            f"N-grams: {NGRAM_RANGE}, min_df={MIN_DOC_FREQ}, max_features={MAX_FEATURES}\n")

    logger.info(f"Saved to: {output_dir}")
    return output_dir
