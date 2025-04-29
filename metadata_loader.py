# metadata_loader.py

import pickle
from bson import decode_file_iter
from snippet_utils import extract_snippet

BSON_PATH     = "web_crawl/pages.bson"
META_OUT_PATH = "doc_meta.pkl"

def build_metadata():
    """
    One-time: scan pages.bson and pickle out
    (doc_id, title, url, snippet) for every document.
    Snippet is meta_description if present,
    otherwise a mid-document sentence from `content`.
    """
    meta = []
    with open(BSON_PATH, "rb") as f:
        for doc in decode_file_iter(f):
            doc_id  = str(doc["_id"])
            title   = doc.get("title") or ""
            url     = doc.get("url") or ""
            desc    = (doc.get("meta_description") or "").strip()
            content = (doc.get("content") or "").strip()

    
            snippet = extract_snippet(content, query=None)

            meta.append((doc_id, title, url, snippet))

    with open(META_OUT_PATH, "wb") as out:
        pickle.dump(meta, out)
    print(f"✔ Pickled metadata for {len(meta)} docs to '{META_OUT_PATH}'")

def load_metadata():
    """
    Load the pickled metadata list:
      [(doc_id, title, url, snippet), ...]
    """
    with open(META_OUT_PATH, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    build_metadata()
