# run_retrieval.py

from retrieval import create_searcher, retrieve, RetrievalError

# Path to your seed queries file (one query per line)
SEED_FILE = "50_queries.txt"
TOP_K = 10  # number of hits to retrieve per query


def load_queries(path: str):
    """
    Reads seed queries from a file, one per line, skipping empty lines.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    queries = load_queries(SEED_FILE)
    print(f"▶ Loaded {len(queries)} queries from '{SEED_FILE}'")

    searcher, parser = create_searcher()
    for q in queries:
        print(f"\n=== Query: {q!r} ===")
        try:
            hits = retrieve(q, top_k=TOP_K, searcher=searcher, parser=parser)
            for rank, h in enumerate(hits, start=1):
                print(f"{rank:2d}. (score={h['score']:.3f}) id={h['doc_id']} title={h['title']}")
        except RetrievalError as e:
            print(f"  [!] RetrievalError: {e}")

if __name__ == "__main__":
    main()
