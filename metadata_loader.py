# metadata_loader.py

import pickle
from bson import decode_file_iter
from snippet_utils import sent_tokenize_regex  # your lightweight regex splitter

BSON_PATH     = "web_crawl/pages.bson"
META_OUT_PATH = "doc_meta.pkl"
MAX_SENTS     = 3   # only first 3 sentences

def build_metadata():
    """
    Stores for each doc: (doc_id, title, url, snippet, [sent1, sent2, sent3])
    """
    meta = []
    with open(BSON_PATH, "rb") as f:
        for doc in decode_file_iter(f):
            doc_id  = str(doc["_id"])
            title   = doc.get("title") or ""
            url     = doc.get("url") or ""
            # short snippet fallback
            desc    = (doc.get("meta_description") or "").strip()
            content = (doc.get("content") or "").strip()

            snippet = desc or (content[:200] + "…")
            sents   = sent_tokenize_regex(content)[:MAX_SENTS]
            meta.append((doc_id, title, url, snippet, sents))

    with open(META_OUT_PATH, "wb") as out:
        pickle.dump(meta, out)
    print(f"✔ Pickled metadata for {len(meta)} docs → '{META_OUT_PATH}'")

def load_metadata():
    with open(META_OUT_PATH, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    build_metadata()
