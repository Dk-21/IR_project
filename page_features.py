# page_features.py

import pickle
from bson import decode_file_iter

BSON_PATH     = "web_crawl/pages.bson"
FEATURES_PATH = "doc_features.pkl"

def build_features():
    """
    Scan pages.bson once and pull out pagerank per doc_id.
    """
    feats = {}
    with open(BSON_PATH, "rb") as f:
        for doc in decode_file_iter(f):
            doc_id = str(doc["_id"])
            feats[doc_id] = float(doc.get("pagerank", 0.0))
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feats, f)
    print(f"✔ Saved pagerank for {len(feats)} docs to '{FEATURES_PATH}'")

def load_features():
    with open(FEATURES_PATH, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    build_features()
