"""
Summary Generator Module (Fast Edition - Flan-T5 Only)
------------------------------------------
職責：
專門負責調用 LLM 生成自然語言摘要。
以確保在無高階 GPU 的環境下也能流暢運行。
"""

import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Hugging Face Imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: 'transformers' library not found. LLM summary will be disabled.")

logger = logging.getLogger(__name__)

_summary_pipeline = None


def rule_based_fallback(aspect_map: dict) -> str:
    """Safe fallback paragraph (no hallucinations).

    The summary is built ONLY from extracted keywords. It prioritizes taste/food first
    and uses frequency to decide emphasis (many/several/some/a few).

    aspect_map schema:
      {"taste": {"pos": [(phrase, freq), ...], "neu": [...], "neg": [...]}, ...}
    """

    def qualifier(freq: int) -> str:
        if freq >= 12:
            return "Many"
        if freq >= 6:
            return "Several"
        if freq >= 3:
            return "Some"
        return "A few"

    def fmt_phrases(items: list[tuple[str, int]], k: int = 2) -> list[str]:
        out = []
        for p, _f in items[:k]:
            s = str(p or "").replace("_", " ").strip()
            if s:
                out.append(s)
        return out

    sentences: list[str] = []

    # 1) Taste first
    taste = aspect_map.get("taste", {}) if isinstance(aspect_map, dict) else {}
    taste_pos = taste.get("pos", []) or []
    taste_neu = taste.get("neu", []) or []
    taste_neg = taste.get("neg", []) or []

    if taste_pos:
        top_freq = int(taste_pos[0][1] or 0)
        q = qualifier(top_freq)
        top_items = fmt_phrases(taste_pos, k=2)
        if len(top_items) == 1:
            sentences.append(f"{q} reviewers praise the {top_items[0]}.")
        else:
            sentences.append(
                f"{q} reviewers praise the {top_items[0]} and {top_items[1]}.")

    # Neutral taste notes (e.g., spicy level) as observations
    if taste_neu:
        items = fmt_phrases(taste_neu, k=2)
        if items:
            sentences.append(f"Taste notes mention {', '.join(items)}.")

    # Taste concerns (only if present)
    if taste_neg:
        top_freq = int(taste_neg[0][1] or 0)
        q = qualifier(top_freq)
        items = fmt_phrases(taste_neg, k=2)
        if items:
            sentences.append(f"{q} mention concerns like {', '.join(items)}.")

    # 2) Service/environment/price/waiting_time next (only what exists)
    for a in ["service", "environment", "price", "waiting_time"]:
        bucket = aspect_map.get(a, {}) if isinstance(aspect_map, dict) else {}
        pos_items = bucket.get("pos", []) or []
        neu_items = bucket.get("neu", []) or []
        neg_items = bucket.get("neg", []) or []

        if pos_items:
            top_freq = int(pos_items[0][1] or 0)
            q = qualifier(top_freq)
            items = fmt_phrases(pos_items, k=2)
            if items:
                label = a.replace("_", " ")
                sentences.append(
                    f"{q} highlight {label} with {', '.join(items)}.")

        if neu_items:
            items = fmt_phrases(neu_items, k=2)
            if items:
                label = a.replace("_", " ")
                sentences.append(
                    f"{label.capitalize()} notes include {', '.join(items)}.")

        if neg_items:
            top_freq = int(neg_items[0][1] or 0)
            q = qualifier(top_freq)
            items = fmt_phrases(neg_items, k=2)
            if items:
                label = a.replace("_", " ")
                sentences.append(
                    f"{q} raise {label} concerns like {', '.join(items)}.")

    # Closing sentence based on presence of positives
    any_pos = any((aspect_map.get(a, {}).get("pos")
                  for a in aspect_map)) if isinstance(aspect_map, dict) else False
    if any_pos:
        sentences.append(
            "Overall, the extracted feedback is more positive than negative.")
    else:
        sentences.append(
            "Overall, the extracted feedback is mixed based on the available keywords.")

    # Keep it short and safe
    return " ".join(sentences[:6]).strip()


def get_summary_pipeline():
    """
    Lazy loading of the summarization pipeline.
    Uses 'google/flan-t5-large' which balances quality and speed on CPUs.
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


def generate_review_summary(input_data, output_root: Path, input_stem: str):
    """
    Generates a natural language summary using constrained Beam Search to prevent hallucinations.
    """
    # Support batch input: a list of place objects (e.g., result.json)
    if isinstance(input_data, list):
        for i, item in enumerate(input_data):
            if not isinstance(item, dict):
                continue
            stem = item.get("name") or item.get(
                "place_name") or f"{input_stem}_{i}"
            generate_review_summary(
                item, output_root=output_root, input_stem=str(stem))
        return

    pipe = None
    if not TRANSFORMERS_AVAILABLE:
        logger.warning(
            "Transformers library missing. Using rule-based fallback.")
    else:
        pipe = get_summary_pipeline()
        if pipe is None:
            logger.warning("Model load failed. Using rule-based fallback.")

    # ----------------------------
    # 1. Data Parsing
    # ----------------------------
    aspect_priority = ["taste", "service",
                       "environment", "price", "waiting_time"]
    aspects_data = {}

    if isinstance(input_data, dict) and "aspects" in input_data:
        # Newer schemas put aspects under a top-level "aspects" key
        aspects_data = input_data.get("aspects", {})
    elif isinstance(input_data, dict):
        # Fallback: treat the dict itself as aspects if it looks like {"taste": {...}, ...}
        # This allows passing just the aspects map directly.
        aspects_data = input_data
    elif isinstance(input_data, pd.DataFrame):
        if input_data.empty:
            return
        for aspect, group in input_data.groupby("aspect"):
            keywords = []
            for _, row in group.iterrows():
                keywords.append({
                    "phrase": row.get("phrase"),
                    "freq": int(row.get("freq", 0)) if pd.notna(row.get("freq", None)) else 0,
                    "sentiment": row.get("sentiment", "neu"),
                    # ignore tfidf for summary purposes
                })
            aspects_data[aspect] = {"keywords": keywords}
    else:
        logger.error("Invalid input data format.")
        return

    # Normalize aspects_data into the canonical shape: {aspect: {"keywords": [..]}}
    normalized = {}
    for aspect, payload in (aspects_data or {}).items():
        if isinstance(payload, dict) and "keywords" in payload:
            kw_list = payload.get("keywords", [])
        elif isinstance(payload, list):
            kw_list = payload
        else:
            kw_list = []

        norm_keywords = []
        for k in kw_list:
            # Allow keywords to be dicts (new format) or raw strings (legacy)
            if isinstance(k, str):
                norm_keywords.append(
                    {"phrase": k, "freq": 1, "sentiment": "neu"})
                continue
            if not isinstance(k, dict):
                continue
            phrase = k.get("phrase")
            if not phrase:
                continue
            freq = k.get("freq", 0)
            try:
                freq = int(freq)
            except Exception:
                freq = 0
            sent = k.get("sentiment", "neu")
            if sent not in {"pos", "neg", "neu"}:
                sent = "neu"
            norm_keywords.append(
                {"phrase": phrase, "freq": freq, "sentiment": sent})

        normalized[str(aspect)] = {"keywords": norm_keywords}

    aspects_data = normalized

    # ----------------------------
    # 2. Build Strict Data Context
    # ----------------------------
    # Build a structured map so we can (a) emphasize by freq and (b) avoid hallucinations.
    # aspect_map: {aspect: {"pos": [(phrase, freq)], "neu": [...], "neg": [...]}}
    aspect_map = {}

    # 排序 Aspect
    existing_keys = list(aspects_data.keys())
    sorted_keys = [a for a in aspect_priority if a in existing_keys]
    sorted_keys += [a for a in existing_keys if a not in aspect_priority]

    for aspect in sorted_keys:
        keywords = aspects_data[aspect].get("keywords", [])
        # 依頻率排序
        keywords.sort(key=lambda x: int(x.get('freq', 0) or 0), reverse=True)

        pos_items: list[tuple[str, int]] = []
        neg_items: list[tuple[str, int]] = []
        neu_items: list[tuple[str, int]] = []

        for k in keywords:
            phrase = str(k.get('phrase', '')).strip()
            if not phrase:
                continue
            phrase = phrase.replace("_", " ").strip()
            freq = int(k.get('freq', 0) or 0)
            sent = k.get('sentiment', 'neu')

            if sent == 'pos':
                pos_items.append((phrase, freq))
            elif sent == 'neg':
                neg_items.append((phrase, freq))
            else:
                neu_items.append((phrase, freq))

        # keep top items per sentiment
        pos_items = pos_items[:5]
        neg_items = neg_items[:4]
        neu_items = neu_items[:4]

        aspect_map[str(aspect)] = {
            "pos": pos_items,
            "neu": neu_items,
            "neg": neg_items,
        }

    # Build STRICT context with frequencies so the model can emphasize correctly.
    data_lines = []
    for aspect in sorted_keys:
        bucket = aspect_map.get(str(aspect), {})

        def _fmt(items: list[tuple[str, int]]) -> str:
            return ", ".join([f"{p}({f})" for p, f in items if p])

        pos_str = _fmt(bucket.get('pos', []))
        neu_str = _fmt(bucket.get('neu', []))
        neg_str = _fmt(bucket.get('neg', []))

        if pos_str:
            data_lines.append(f"{str(aspect).upper()}_POS: {pos_str}")
        if neu_str:
            data_lines.append(f"{str(aspect).upper()}_NEU: {neu_str}")
        if neg_str:
            data_lines.append(f"{str(aspect).upper()}_NEG: {neg_str}")

    context_str = "\n".join(data_lines)

    # ----------------------------
    # 3. Prompt Engineering (Strict Mode)
    # ----------------------------
    # T5 喜歡明確的 Task 定義。這裡我們移除 "Yelp Reviewer" 的角色扮演。
    prompt = (
        "You are summarizing restaurant review keywords. "
        "Use ONLY the information in the Data block; do not add or infer any details (no location/indoors/outdoors/noise/price range) unless explicitly present. "
        "Prioritize TASTE first: start the summary with food quality or taste. "
        "Use the numbers in parentheses as frequency counts to control emphasis (higher counts => stronger wording like 'many'/'often'; lower counts => 'some'/'a few'). "
        "Then mention service/environment/price/waiting_time ONLY if they appear in the data. "
        "Keep it to 4–6 sentences, natural American English, and avoid repetition.\n\n"
        f"Data:\n{context_str}\n\n"
        "Summary:"
    )

    # ----------------------------
    # 4. Generation (Beam Search)
    # ----------------------------
    try:
        logger.info("Generating summary...")

        if pipe is None:
            summary_text = rule_based_fallback(aspect_map)
            # Save Logic
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = output_root / input_stem if input_stem else output_root
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"summary_{ts}.txt"

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(summary_text)

            logger.info(f"Summary saved to {out_path}")
            print("\n--- Generated Summary (Rule-based Fallback) ---")
            print(summary_text)
            print("----------------------------------------------")
            return

        output = pipe(
            prompt,
            max_length=300,
            min_length=80,       # 強制長度，避免只吐出一句話
            do_sample=False,     # 關閉隨機採樣，避免幻覺
            num_beams=5,         # 開啟 Beam Search (尋找最佳路徑)
            no_repeat_ngram_size=3,  # 避免重複
            early_stopping=True
        )

        summary_text = output[0]['generated_text'].replace(
            "Summary:", "").strip()

        # Save Logic
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = output_root / input_stem if input_stem else output_root
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"summary_{ts}.txt"

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        logger.info(f"Summary saved to {out_path}")
        print("\n-------- Generated Summary --------")
        print(summary_text)
        print("------------------------------------")

    except Exception as e:
        logger.exception(f"Failed to generate summary: {e}")
