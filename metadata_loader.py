# metadata_loader.py

import pickle
from bson import decode_file_iter

BSON_PATH = "web_crawl/pages.bson"
META_PATH = "doc_meta.pkl"   # will store List[(doc_id, title, url)]

# Helper to pick text for filtering
import re
from html import unescape

def strip_html(html: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', html or "")
    return unescape(text).strip()

def pick_text(doc: dict) -> str:
    # Prefer content, then meta_description, else fallback to stripped html
    for field in ("content", "meta_description"):
        val = doc.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return strip_html(doc.get("html_content"))


def build_metadata():
    """
    Reads pages.bson, extracts (doc_id, title, url) for only "good" docs:
    - Must have a non-empty TF-IDF dict (so it matches the matrix builder)
    - Must have some usable text (content, meta_description, or html_content)
    Saves list in META_PATH.
    """
    meta = []
    with open(BSON_PATH, "rb") as f:
        for doc in decode_file_iter(f):
            # Skip if no TF-IDF data
            tf = doc.get("tfidf") or {}
            if not tf:
                continue

            # Extract and sanitize fields
            doc_id = str(doc.get("_id"))
            title  = (doc.get("title") or "").strip()
            url    = (doc.get("url") or "").strip()

            # Ensure there's text to show
            snippet = pick_text(doc)
            if not snippet:
                continue

            meta.append((doc_id, title, url))

    # Persist metadata
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)
    print(f"✔ Saved metadata for {len(meta)} docs to '{META_PATH}'")
    return meta


def load_metadata():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    build_metadata()
