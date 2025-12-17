# main CLI entry for SpotLite
import argparse
import logging
from pathlib import Path
import json

# Import the new AspectKeywordAnalyzer
try:
    from spotlite.analysis.keywords import AspectKeywordAnalyzer
except ImportError:
    AspectKeywordAnalyzer = None
    print("❌ Warning: AspectKeywordAnalyzer not found.")

# Import the summary generator independently
try:
    from spotlite.analysis.summary import generate_review_summary
except ImportError:
    generate_review_summary = None
    print("❌ Warning: summary_generator not found.")

from spotlite.config.config import load_config
from spotlite.config.logging_config import setup_logging
from spotlite.crawler.google_maps.reviews import scrape_reviews_for_url, save_reviews
from spotlite.crawler.google_maps.details import get_place_details, get_place_id, save_details

API_KEYS = load_config("api_keys.json")
logger = logging.getLogger(__name__)


# ---------- Subcommand handlers ----------

def cmd_reviews(args):
    url = args.url
    place_name, reviews = scrape_reviews_for_url(url)
    save_reviews(place_name, reviews)


def cmd_details(args):
    url = args.url
    google_cfg = API_KEYS.get("google_maps", {})
    api_key = google_cfg.get("api_key")
    if not api_key:
        logger.error("❌ google_maps.api_key not set in configs/api_keys.json")
        return

    place_id = get_place_id(api_key, url)
    if not place_id:
        logger.error("❌ Could not extract place_id from URL")
        return

    result = get_place_details(api_key, place_id)
    save_details(result, place_id=place_id)


def cmd_analyze(args):
    """
    [Phase 1] 關鍵字提取與數據分析
    - 讀取原始 reviews.json
    - 進行 NLP 處理、TF-IDF 計算
    - 輸出 CSV 報表
    """
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("❌ Input file does not exist: %s", input_path)
        return

    if AspectKeywordAnalyzer is None:
        logger.error("❌ AspectKeywordAnalyzer class is missing.")
        return

    logger.info(f"🔍 Start Keyword Analysis on {input_path.name}")

    try:
        analyzer = AspectKeywordAnalyzer(domain=args.domain)

        # 1. Load Data
        analyzer.load_data(input_path)

        # 2. Extract Phrases & Sentiment
        analyzer.extract_phrases()

        # 3. Compute Stats
        analyzer.compute_tfidf()

        # 4. Save CSV Results (Do NOT generate summary here)
        analyzer.save_results()

        logger.info("✅ Keyword Analysis finished. CSVs saved.")

    except Exception as e:
        logger.exception(f"❌ Analysis failed: {e}")


def cmd_summary(args):
    """
    [Phase 2] LLM 摘要生成
    - 讀取 Phase 1 產生的 top5_keywords.json
    - 調用 LLM 生成自然語言摘要
    """
    input_path = Path(args.input)

    if not input_path.exists():
        logger.error("❌ Input file does not exist: %s", input_path)
        return

    # 強制檢查 JSON
    if input_path.suffix.lower() != '.json':
        logger.error("❌ Invalid format. Please provide a .json file.")
        return

    if generate_review_summary is None:
        logger.error("❌ Summary generator module is missing.")
        return

    logger.info(f"📝 Start Summary Generation from {input_path.name}")

    try:
        # [MODIFIED] 使用標準 json lib 讀取為 Dictionary
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Determine output root
        output_dir = input_path.parent
        input_stem = ""

        # Call the generator with Dictionary
        generate_review_summary(data, output_dir, input_stem)

        logger.info("✅ Summary Generation finished.")

    except Exception as e:
        logger.exception(f"❌ Summary generation failed: {e}")


# ---------- Main CLI ----------

def build_parser():
    parser = argparse.ArgumentParser(
        description="SpotLite CLI - Google Maps details/reviews & analysis"
    )
    parser.add_argument("-d", "--debug", action="store_true",
                        help="Enable debug logging and save logs to file")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # reviews
    p_reviews = subparsers.add_parser(
        "reviews", help="Scrape Google Maps reviews")
    p_reviews.add_argument("-u", "--url", required=True,
                           help="Google Maps place URL")
    p_reviews.set_defaults(func=cmd_reviews)

    # details
    p_details = subparsers.add_parser(
        "details", help="Fetch Google Maps place details via Places API")
    p_details.add_argument("-u", "--url", required=True,
                           help="Google Maps place URL")
    p_details.set_defaults(func=cmd_details)

    # analyze (Phase 1)
    p_analyze = subparsers.add_parser(
        "kewords", help="Phase 1: Extract Keywords & Sentiment (Outputs CSV)")
    p_analyze.add_argument("-i", "--input", required=True,
                           help="Path to raw reviews.json")
    p_analyze.add_argument("--domain", default="restaurant",
                           help="Domain name (e.g. restaurant)")
    p_analyze.set_defaults(func=cmd_analyze)

    # summary (Phase 2)
    p_summary = subparsers.add_parser(
        "summary", help="Phase 2: Generate LLM Summary from CSV")
    p_summary.add_argument("-i", "--input", required=True,
                           help="Path to top5_keywords_xxx.csv generated by analyze command")
    p_summary.set_defaults(func=cmd_summary)

    return parser


def main():
    parser = build_parser()
    try:
        args = parser.parse_intermixed_args()
    except Exception:
        args = parser.parse_args()

    setup_logging(force_debug=args.debug)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled via -d flag")

    # Dispatch to the selected subcommand
    args.func(args)


if __name__ == "__main__":
    main()
