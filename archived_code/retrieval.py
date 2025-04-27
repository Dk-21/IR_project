# retrieval.py

from whoosh.index import open_dir
from whoosh.qparser import MultifieldParser, QueryParserError
from whoosh import scoring

INDEX_DIR     = "whoosh_index"
SEARCH_FIELDS = ["title", "content"]

class RetrievalError(Exception):
    pass

def create_searcher(index_dir: str = INDEX_DIR):
    ix = open_dir(index_dir)
    return ix.searcher(weighting=scoring.TF_IDF()), MultifieldParser(SEARCH_FIELDS, schema=ix.schema)

def retrieve(query_str: str, top_k: int = 10, searcher=None, parser=None):
    if searcher is None or parser is None:
        searcher, parser = create_searcher()
    try:
        q = parser.parse(query_str)
    except QueryParserError as e:
        raise RetrievalError(f"Parse error: {e}")
    results = searcher.search(q, limit=top_k)
    if not results:
        raise RetrievalError(f"No results for {query_str!r}")
    return [
        {
            "doc_id": hit["id"],
            "title":  hit.get("title",""),
            "url":    hit.get("url",""),
            "score":  hit.score
        }
        for hit in results
    ]

if __name__ == "__main__":
    s,p = create_searcher()
    for q in ["marathon training", "best running shoes"]:
        try:
            hits = retrieve(q, top_k=5, searcher=s, parser=p)
            print(q, [h["title"] for h in hits])
        except RetrievalError as e:
            print("Error:", e)
