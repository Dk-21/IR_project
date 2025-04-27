# run_feedback_vec.py

import csv
from retrieval_vec import retrieve_vec
from feeback_vec import retrieve_expanded

SEED_FILE   = "50_queries.txt"
RESULTS_CSV = "feedback_vec_results.csv"
TOP_K       = 10


def load_queries(path: str):
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def main():
    queries = load_queries(SEED_FILE)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query","baseline","expanded_query","expanded"])

        for q in queries:
            base_hits = retrieve_vec(q, top_k=TOP_K)
            base_entries = [f"{h['title']}|{h['url']}" for h in base_hits]

            exp_q, exp_hits = retrieve_expanded(q)
            exp_entries = [f"{h['title']}|{h['url']}" for h in exp_hits]

            w.writerow([
                q,
                ";".join(base_entries),
                exp_q,
                ";".join(exp_entries)
            ])
            print(f"Processed: {q!r}")

    print(f"✔ Results written to '{RESULTS_CSV}'")

if __name__=="__main__":
    main()