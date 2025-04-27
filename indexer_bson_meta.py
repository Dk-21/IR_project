# indexer_bson_meta.py

import os, shutil, sys, re
from html import unescape
from bson import decode_file_iter
from whoosh.index import create_in
from whoosh.fields import Schema, ID, TEXT, BOOLEAN, NUMERIC
from whoosh.analysis import StemmingAnalyzer

BSON_PATH = "web_crawl/pages.bson"
INDEX_DIR = "whoosh_index"

def strip_html(html: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', html or "")
    return unescape(text).strip()

def pick_text(doc: dict) -> str:
    for field in ("content", "meta_description"):
        val = doc.get(field)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return strip_html(doc.get("html_content", ""))

def build_index():
    # Remove old index
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    os.makedirs(INDEX_DIR)

    # Define schema with metadata
    schema = Schema(
        id             = ID(stored=True, unique=True),
        title          = TEXT(stored=True, analyzer=StemmingAnalyzer()),
        url            = ID(stored=True),
        content        = TEXT(analyzer=StemmingAnalyzer()),
        tfidf_updated  = BOOLEAN(stored=True),
        auth_score     = NUMERIC(stored=True, decimal_places=4),
        hub_score      = NUMERIC(stored=True, decimal_places=4),
        pagerank       = NUMERIC(stored=True, decimal_places=6),
        topic_pagerank = NUMERIC(stored=True, decimal_places=6),
    )

    ix = create_in(INDEX_DIR, schema)
    writer = ix.writer(limitmb=2048)
    total = 0

    with open(BSON_PATH, "rb") as f:
        for doc in decode_file_iter(f):
            doc_id = str(doc.get("_id"))
            text   = pick_text(doc)
            if not text:
                continue

            # Extract metadata with defaults
            tfu = bool(doc.get("tfidf_updated", False))
            auth = float(doc.get("auth_score") or 0.0)
            hub  = float(doc.get("hub_score")  or 0.0)
            pr   = float(doc.get("pagerank")   or 0.0)
            tpr  = float(doc.get("topic_pagerank") or 0.0)

            writer.add_document(
                id             = doc_id,
                title          = doc.get("title","") or "",
                url            = doc.get("url","") or "",
                content        = text,
                tfidf_updated  = tfu,
                auth_score     = auth,
                hub_score      = hub,
                pagerank       = pr,
                topic_pagerank = tpr
            )
            total += 1
            if total % 10000 == 0:
                print(f"  Indexed {total} docs…")

    writer.commit()
    print(f"✅ Indexed {total} documents with metadata into '{INDEX_DIR}'")

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print("ERROR during indexing:", e)
        sys.exit(1)
