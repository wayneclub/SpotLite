
from pathlib import Path
import re
from spotlite.utils.io_utils import load_json

URL_RE = re.compile(r"http\S+|www\.\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")


def load_reviews_json(path: Path):
    """Load reviews from a JSON file, returning a list of cleaned text reviews."""
    data = load_json(path)
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
