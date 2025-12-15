# spotlite/analysis/aspect_phrases.py
"""
AspectPhraseExtractor

這個模組只負責一件事：從「已經清理好 / 統一 schema 的 reviews」
中抽出 (aspect, sentiment) 關鍵詞，並用 TF‑IDF + 聚合整理。

它**不**處理：
- 怎麼從 Google Maps / Yelp 抓資料
- 怎麼把 raw JSON 轉成 unified schema
- 怎麼做結構化 review parsing（那是 preprocess 層在做的事）

使用方式（給 Base / GoogleMapsAnalyzer 調用）大致會是：

    extractor = AspectPhraseExtractor(domain="restaurant")
    outputs = extractor.run(
        reviews=plain_reviews,          # List[dict]，每則含 text / rating
        place_name=meta.get("name"),
        output_dir=Path("outputs/.."),
    )

    tfidf_df   = outputs["tfidf"]
    agg_df     = outputs["aggregated_targets"]

其他模組只要把 reviews 準備好丟進來就行。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter

import logging
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from spotlite.config.aspect_config import init_domain, DOMAIN_CACHE
from spotlite.analysis.aspect_core import (
    ASPECT_EMB_MODEL,
    clean_review_text,
    extract_adj_noun_phrases,
    assign_aspect,
    rating_to_sentiment,
    phrase_sentiment,
)

logger = logging.getLogger(__name__)


class AspectPhraseExtractor:
    """
    負責從 reviews 中抽出 (aspect, sentiment) 關鍵詞，並用 TF‑IDF + 聚合整理。

    期望輸入：
        reviews: List[dict]
            每一筆為 unified review record，至少包含：
                - "text": 原始評論文字
                - "rating": (可選) 數字星等 1–5

    回傳：
        {
          "tfidf": DataFrame,             # 每個 aspect/sentiment/phrase 的 TF‑IDF 分數
          "aggregated_targets": DataFrame # 以名詞為中心聚合後的結果
        }
    """

    def __init__(self, domain: str = "restaurant") -> None:
        """
        :param domain: e.g. 'restaurant', 'hotel' ...
        會透過 init_domain(domain) 載入該領域的：
          - STOPWORDS
          - PROTECTED_PHRASES
          - ASPECT_SEEDS
        真正的全域變數都在 aspect_core / aspect_config 裡維護。
        """
        self.domain = domain
        # 先由 aspect_config 載入該 domain 的設定，結果會存進 DOMAIN_CACHE
        init_domain(domain)
        domain_cfg = DOMAIN_CACHE[domain]

        # protected_phrases: Dict[str, str]，例如 {"kids meal": "kids_meal", ...}
        self.protected_map = domain_cfg["protected_phrases"]

        # aspect_seeds: 例如 {"taste": {"pos": [...], "neg": [...]}, ...}
        self.aspect_seeds = domain_cfg["aspect_seeds"]

        # optional sentiment seeds precomputed per domain
        domain_sent = domain_cfg.get("sentiment_seeds") or {}

        def _norm_seed(x: object) -> str:
            # Match phrase surface form: lowercase + underscores
            return str(x).lower().strip().replace(" ", "_")

        self.seeds_pos_global = [_norm_seed(w)
                                 for w in domain_sent.get("pos", [])]
        self.seeds_neg_global = [_norm_seed(w)
                                 for w in domain_sent.get("neg", [])]

        # sentence embedding 模型（目前在 aspect_core 裡初始化 ASPECT_EMB_MODEL）
        self.model = ASPECT_EMB_MODEL
        # similarity threshold for assigning aspect by embedding similarity
        self.sim_threshold: float = 0.35

        self.aspect_seed_embs = {}
        for aspect, cfg in self.aspect_seeds.items():
            if isinstance(cfg, dict):
                seed_words = cfg.get("pos", []) + cfg.get("neg", [])
            else:
                seed_words = list(cfg)
            seed_words = [w.lower() for w in seed_words]
            if not seed_words:
                continue
            emb = self.model.encode(
                seed_words,
                convert_to_tensor=True,
                show_progress_bar=True,
            )
            self.aspect_seed_embs[aspect] = (seed_words, emb)

        # sentiment seeds（跨所有 aspect 合併正向 / 負向清單，用來判斷 phrase 正負）
        # 如果 config 裡沒有額外的 sentiment_seeds，就從各個 aspect 的 seeds 裡合併一份全域正負詞彙
        if not self.seeds_pos_global and not self.seeds_neg_global:
            pos_set = set()
            neg_set = set()
            for aspect, seed_cfg in self.aspect_seeds.items():
                # 支援兩種格式：
                # 1) {"taste": {"pos": [...], "neg": [...]}}
                # 2) {"taste": ["fresh", "delicious", "tasty", ...]} （舊版）
                if isinstance(seed_cfg, dict):
                    # Backward-compatible support:
                    # - new schema: {"pos": [...], "neg": [...]}
                    # - legacy schema: {"seeds_pos": [...], "seeds_neg": [...]}
                    pos_list = (
                        seed_cfg.get("pos")
                        or seed_cfg.get("seeds_pos")
                        or []
                    )
                    neg_list = (
                        seed_cfg.get("neg")
                        or seed_cfg.get("seeds_neg")
                        or []
                    )
                    for w in pos_list:
                        pos_set.add(_norm_seed(w))
                    for w in neg_list:
                        neg_set.add(_norm_seed(w))
                elif isinstance(seed_cfg, (list, tuple)):
                    # 舊版只給一串 keywords，全部當成正面 seeds
                    for w in seed_cfg:
                        pos_set.add(_norm_seed(w))
                else:
                    # 其他型別就忽略
                    continue
            self.sent_seeds_pos = sorted(pos_set)
            self.sent_seeds_neg = sorted(neg_set)
        else:
            # 已經從 config 讀到 sentiment_seeds，就直接使用
            self.sent_seeds_pos = self.seeds_pos_global
            self.sent_seeds_neg = self.seeds_neg_global

        logger.info(
            "Sentiment seeds loaded: pos=%d, neg=%d (neg sample=%s)",
            len(self.sent_seeds_pos),
            len(self.sent_seeds_neg),
            self.sent_seeds_neg[:10],
        )

        # Token-level seed sets (underscore-split) to improve recall on canonical phrases
        STOP_SEED_TOKENS = {"a", "an", "the", "and", "or", "to",
                            "of", "for", "in", "on", "at", "with", "is", "was", "are"}
        self.sent_seed_tokens_pos = set()
        self.sent_seed_tokens_neg = set()

        for s in self.sent_seeds_pos:
            for tok in str(s).split("_"):
                tok = tok.strip()
                if tok and tok not in STOP_SEED_TOKENS:
                    self.sent_seed_tokens_pos.add(tok)

        for s in self.sent_seeds_neg:
            for tok in str(s).split("_"):
                tok = tok.strip()
                if tok and tok not in STOP_SEED_TOKENS:
                    self.sent_seed_tokens_neg.add(tok)

        # Common negation tokens (helps "not_good_*" or "no_*" patterns)
        self.sent_seed_tokens_neg.update(
            {"not", "no", "never", "hardly", "barely", "cannot", "cant",
                "won't", "wont", "dont", "didnt", "isnt", "wasnt", "arent"}
        )

        logger.info(
            "Sentiment seed tokens built: pos=%d, neg=%d (neg token sample=%s)",
            len(self.sent_seed_tokens_pos),
            len(self.sent_seed_tokens_neg),
            sorted(list(self.sent_seed_tokens_neg))[:15],
        )

    # ------------------------------------------------------------------
    # 1. 將 raw reviews 整理成內部 record 形式
    # ------------------------------------------------------------------
    def _build_records(self, raw_reviews: List[dict]) -> List[dict]:
        """
        把 raw reviews 整理成 internal record:
            { "raw_text": str, "cleaned": str, "rating": Optional[float] }

        - 支援幾種常見欄位名稱：text / reviewText / content / body / review
        - cleaned 會使用 clean_review_text 做小寫化 / URL 清除 / 保留 protected phrases 等。
        """
        records: List[dict] = []
        for item in raw_reviews:
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
                # 不認得的型別就跳過
                continue

            if not text or not text.strip():
                continue
            cleaned = clean_review_text(text, self.protected_map)
            if not cleaned:
                continue

            records.append(
                {
                    "raw_text": text,
                    "cleaned": cleaned,
                    "rating": rating,
                }
            )

        logger.info("Built %d cleaned records from raw reviews", len(records))
        return records

    # ------------------------------------------------------------------
    # 2. 抽出 aspect phrases 並依 (aspect, sentiment) 分組成「文件」
    # ------------------------------------------------------------------
    def _collect_phrase_docs(
        self,
        records: List[dict],
    ) -> Tuple[Dict[Tuple[str, str], List[str]], Counter]:
        """
        Step 2~3：對每則 review：

          - 用 extract_adj_noun_phrases 抽出 phrase（已經是 protected_phrases 格式）
          - 用 assign_aspect 指定 taste / service / environment / price / waiting_time
          - 用 phrase_sentiment 判斷正負；若無法判斷則 fallback 到 review 等級星等

        回傳：
            grouped_docs[(aspect, sentiment)] = [
                "phrase1 phrase2 ...",   # 每一則 review 對應一個「小文件」
                ...
            ]

            phrase_stats[(aspect, sentiment, phrase)] = 出現次數
        """
        grouped_docs: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        phrase_stats: Counter = Counter()

        for rec in records:
            rating = rec["rating"]
            review_level_sent = rating_to_sentiment(rating)

            text_clean = rec["cleaned"]
            phrases = extract_adj_noun_phrases(text_clean)
            if not phrases:
                continue

            # 臨時 map，避免同一則 review 內過度重複 push
            aspect_sent_map: Dict[Tuple[str, str],
                                  List[str]] = defaultdict(list)

            for ph in phrases:
                # Phrase is already canonicalized by extract_adj_noun_phrases
                canon = ph

                aspect = assign_aspect(
                    phrase=canon.replace("_", " "),
                    model=self.model,
                    aspect_seed_embs=self.aspect_seed_embs,
                    sim_threshold=self.sim_threshold,
                )
                if not aspect:
                    continue

                # Determine sentiment at phrase level (NEG-first, token-aware)
                sent = None
                parts = canon.split("_")

                # 1) Token-level NEG-first check (best recall on canonical phrases)
                if any(t in self.sent_seed_tokens_neg for t in parts):
                    sent = "neg"
                elif any(t in self.sent_seed_tokens_pos for t in parts):
                    sent = "pos"

                # 2) Backward-compatible substring check using full seeds list
                if sent is None:
                    sent = phrase_sentiment(
                        canon,
                        self.sent_seeds_pos,
                        self.sent_seeds_neg,
                    )

                # 3) Canonical-aware adjective polarity (works even if seeds are multi-word)
                if sent is None and parts:
                    adj = parts[0]
                    if adj in self.sent_seed_tokens_neg or adj in self.sent_seeds_neg:
                        sent = "neg"
                    elif adj in self.sent_seed_tokens_pos or adj in self.sent_seeds_pos:
                        sent = "pos"

                # Final fallback
                if sent is None:
                    sent = review_level_sent or "neutral"

                # Prevent neutral dominance: only keep neutral if no pos/neg exists for this review+aspect
                if sent == "neutral":
                    continue

                key = (aspect, sent)
                aspect_sent_map[key].append(canon)

            # 把這一則 review 的 phrases 合併成一個「文件」
            for (aspect, sent), ph_list in aspect_sent_map.items():
                if not ph_list:
                    continue
                key = (aspect, sent)
                doc_str = " ".join(ph_list)
                grouped_docs[key].append(doc_str)
                for canon in ph_list:
                    phrase_stats[(aspect, sent, canon)] += 1

        return grouped_docs, phrase_stats

    # ------------------------------------------------------------------
    # 3. 在每個 (aspect, sentiment) 子集合上跑 TF‑IDF
    # ------------------------------------------------------------------
    def _run_tfidf(
        self,
        grouped_docs: Dict[Tuple[str, str], List[str]],
        phrase_stats: Counter,
    ) -> pd.DataFrame:
        """
        Step 4：在每個 (aspect, sentiment) 子集合上跑 TF‑IDF。
        每個 key 對應一個獨立的 TF‑IDF 模型，最後再合併成一張表。
        """
        rows_tfidf: List[Dict] = []

        for (aspect, sentiment), docs in grouped_docs.items():
            docs_clean = [d for d in docs if isinstance(d, str) and d.strip()]
            n_docs = len(docs_clean)
            if n_docs < 1:
                continue

            # 依照文件數量調整 min_df（ngram 固定為 (1,1)，確保每個 canonical phrase 作為單一 token）
            if n_docs <= 10:
                min_df = 1
            elif n_docs <= 50:
                min_df = 2
            else:
                min_df = max(2, int(0.01 * n_docs))

            logger.info(
                "TF-IDF for aspect=%s, sentiment=%s: n_docs=%d, ngram_range=%s, min_df=%s",
                aspect,
                sentiment,
                n_docs,
                (1, 1),
                min_df,
            )

            # Use canonical-phrase-safe vectorizer: each phrase is a single token, no re-tokenization
            vec = TfidfVectorizer(
                tokenizer=lambda s: s.split(),      # phrases are already space-separated tokens
                preprocessor=lambda s: s,           # no additional preprocessing
                token_pattern=None,                 # disable internal regex tokenization
                # each canonical phrase is one token
                ngram_range=(1, 1),
                min_df=min_df,
            )

            try:
                X = vec.fit_transform(docs_clean)
            except ValueError as exc:
                logger.warning(
                    (
                        "Skipping TF-IDF for aspect=%s, sentiment=%s "
                        "(n_docs=%d, ngram_range=%s, min_df=%s) due to error: %s"
                    ),
                    aspect,
                    sentiment,
                    n_docs,
                    (1, 1),
                    min_df,
                    exc,
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
            return pd.DataFrame()

        return pd.DataFrame(rows_tfidf)

    # ------------------------------------------------------------------
    # 4. 依 noun 聚合形容詞，產出 aspect_targets_aggregated
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate_targets(tfidf_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 5：依 noun 聚合形容詞，產出 aspect_targets_aggregated。

        phrase 目前假設大多為 "adj_noun" 形式，例如：
            fresh_seafood, delicious_taco

        會被拆成：
            noun  = seafood / taco
            adj   = fresh / delicious

        並統計每個 noun 的形容詞出現次數，作為之後 AI summary 的輸入。
        """
        rows: List[Dict] = []
        if tfidf_df.empty:
            return pd.DataFrame()

        for (aspect, sentiment), sub_df in tfidf_df.groupby(["aspect", "sentiment"]):
            target_adj_counts: Dict[str, Counter] = defaultdict(Counter)
            for _, row in sub_df.iterrows():
                phrase = row["phrase"]
                freq = int(row.get("freq", 0))
                parts = phrase.split("_", 1)
                if len(parts) != 2:
                    continue
                adj, noun = parts[0], parts[1]
                target_adj_counts[noun][adj] += max(freq, 1)

            for noun, adj_counter in target_adj_counts.items():
                total_mentions = int(sum(adj_counter.values()))
                top_adjs = [
                    f"{adj}({cnt})" for adj, cnt in adj_counter.most_common(5)
                ]
                rows.append(
                    {
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "target_noun": noun,
                        "total_mentions": total_mentions,
                        "top_adjectives": ", ".join(top_adjs),
                    }
                )

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows).sort_values(
            ["aspect", "sentiment", "total_mentions"],
            ascending=[True, True, False],
        )

    # ------------------------------------------------------------------
    # 5. 對外主入口
    # ------------------------------------------------------------------
    def run(
        self,
        reviews: List[dict],
    ) -> Dict[str, object]:
        """
        主入口：給 unified 的 reviews List[dict]，回傳 TF‑IDF 與聚合結果。

        Args:
            reviews:
                List[dict]，每一筆至少要有 "text" 欄位，可選 "rating"。

        Returns:
            dict with keys:
                - "tfidf": pd.DataFrame
                - "aggregated_targets": pd.DataFrame
        """
        records = self._build_records(reviews)
        grouped_docs, phrase_stats = self._collect_phrase_docs(records)
        if not grouped_docs:
            logger.warning("No aspect phrases extracted; check seeds / data.")
            return {
                "tfidf": pd.DataFrame(),
                "aggregated_targets": pd.DataFrame(),
            }

        tfidf_df = self._run_tfidf(grouped_docs, phrase_stats)
        agg_df = self._aggregate_targets(tfidf_df)

        return {
            "tfidf": tfidf_df,
            "aggregated_targets": agg_df,
        }
