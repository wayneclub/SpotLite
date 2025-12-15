# main CLI entry for SpotLite
import argparse
import logging
from pathlib import Path

from spotlite.analysis.google_maps_analyzer import GoogleMapsAnalyzer
from spotlite.config.config import load_config
from spotlite.config.logging_config import setup_logging
from spotlite.crawler.google_maps.reviews import scrape_reviews_for_url, save_reviews
from spotlite.crawler.google_maps.details import get_place_details, get_place_id, save_details
from spotlite.analysis.keywords import analyze_keywords
from spotlite.utils.io_utils import load_json


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
    使用 GoogleMapsAnalyzer 對「未處理過的 reviews.json」做 NLP 分析，
    根據 domain 產生 CSV (TF-IDF + aggregated targets)。
    """
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("❌ Input file does not exist: %s", input_path)
        return

    logger.info("🔍 Start analysis")

    analyzer = GoogleMapsAnalyzer(domain=args.domain)

    raw_json = load_json(input_path)   # 讀取爬取後的 JSON
    result = analyzer.run_analysis(raw_json)
    logger.info("✅ Analysis finished.")


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

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze reviews JSON (NLP, keywords, etc.)")
    p_analyze.add_argument("-i", "--input", required=True,
                           help="Path to reviews.json")
    p_analyze.add_argument("--domain", required=True,
                           help="Domain name (e.g. restaurant, hotel, airline)")
    p_analyze.set_defaults(func=cmd_analyze)

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
