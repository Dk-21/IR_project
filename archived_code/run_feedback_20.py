# run_feedback_20.py

import os
import csv
from retrieval import create_searcher, retrieve, RetrievalError
from feedback import rocchio_expand

# Path to your seed queries file (one query per line)
SEED_FILE = "50_queries.txt"
# File to write results (will be overwritten each run)
RESULTS_FILE = "feedback_20_results.csv"
TOP_K = 10  # number of hits to retrieve for baseline and expanded


def load_queries(path: str):
    """
    Reads seed queries from a file, one per line, skipping empty lines.
    """
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    # Load seed queries
    queries = load_queries(SEED_FILE)
    print(f"▶ Loaded {len(queries)} queries from '{SEED_FILE}'")

    # Initialize results file (overwrite with header)
    with open(RESULTS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query", 
            "baseline_results", 
            "expanded_query", 
            "expanded_results"
        ])

    # Prepare Whoosh searcher
    searcher, parser = create_searcher()

    for q in queries:
        print(f"\n===\nQuery: {q!r}")

        # 1) Baseline retrieval
        try:
            baseline_hits = retrieve(q, top_k=TOP_K, searcher=searcher, parser=parser)
            baseline_entries = []
            print(" Baseline results:")
            for rank, h in enumerate(baseline_hits, start=1):
                entry = f"{h['title']}|{h['url']}"
                baseline_entries.append(entry)
                print(f"  {rank:2d}. title={h['title']!r} url={h['url']}")
        except RetrievalError as e:
            print(" [!] Baseline retrieval error:", e)
            baseline_entries = []

        # 2) Rocchio expansion
        try:
            exp_q = rocchio_expand(q)
            print(" Expanded query:", exp_q)
        except Exception as e:
            print(" [!] Expansion error:", e)
            exp_q = ""

        # 3) Retrieval on expanded query
        try:
            expanded_hits = retrieve(exp_q, top_k=TOP_K, searcher=searcher, parser=parser)
            expanded_entries = []
            print(" Expanded results:")
            for rank, h in enumerate(expanded_hits, start=1):
                entry = f"{h['title']}|{h['url']}"
                expanded_entries.append(entry)
                print(f"  {rank:2d}. title={h['title']!r} url={h['url']}")
        except RetrievalError as e:
            print(" [!] Expanded retrieval error:", e)
            expanded_entries = []

        # Append results row
        if len(expanded_entries)>0:
            with open(RESULTS_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    q,
                    ";".join(baseline_entries),
                    exp_q,
                    ";".join(expanded_entries)
                ])

    print(f"✔ Results written to '{RESULTS_FILE}' (overwritten each run)")

if __name__ == "__main__":
    main()
