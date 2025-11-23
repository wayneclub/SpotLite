import json
from pathlib import Path
import re


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
