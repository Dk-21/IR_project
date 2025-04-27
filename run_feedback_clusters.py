# run_feedback_clusters.py

import csv, numpy as np, pickle
from retrieval import create_searcher, retrieve, RetrievalError
from bson_loader import load_corpus_matrix
from clusters import association_clusters, metric_clusters, scalar_clusters
from feedback import TOP_K, NR_K, ALPHA, BETA, GAMMA, EXPANSION_TERMS

SEED_FILE  = "50_queries.txt"
OUTPUT_CSV = "clustered_feedback.csv"

# Load TF-IDF matrix & doc IDs
tfidf_matrix, doc_ids = load_corpus_matrix()
with open("tfidf_vectorizer.pkl","rb") as f:
    vectorizer = pickle.load(f)
feature_names = vectorizer.get_feature_names_out()
doc_to_idx   = {d:i for i,d in enumerate(doc_ids)}

# Whoosh
searcher, parser = create_searcher()

def load_queries(path):
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

def main():
    queries = load_queries(SEED_FILE)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
        w = csv.writer(fout)
        w.writerow(["query","assoc_q","metric_q","scalar_q"])
        for q in queries:
            try:
                hits = retrieve(q, top_k=TOP_K+NR_K, searcher=searcher, parser=parser)
            except RetrievalError:
                continue
            rel, nonrel = hits[:TOP_K], hits[TOP_K:TOP_K+NR_K]
            q_vec = vectorizer.transform([q]).toarray()[0]
            R = tfidf_matrix[[doc_to_idx[h["doc_id"]] for h in rel]].toarray().mean(0)
            N = tfidf_matrix[[doc_to_idx[h["doc_id"]] for h in nonrel]].toarray().mean(0)
            q_mod = ALPHA*q_vec + BETA*R - GAMMA*N
            topi = np.argsort(q_mod)[-EXPANSION_TERMS:]
            candidates = feature_names[topi]

            assoc  = association_clusters(list(candidates), [doc_to_idx[h["doc_id"]] for h in rel], tfidf_matrix, feature_names)
            metric = metric_clusters(   list(candidates), [doc_to_idx[h["doc_id"]] for h in rel], tfidf_matrix, feature_names)
            scalar = scalar_clusters(   list(candidates), vectorizer)

            w.writerow([
                q,
                q + " " + " ".join(assoc),
                q + " " + " ".join(metric),
                q + " " + " ".join(scalar)
            ])
            print(f"Processed: {q!r}")

    print(f"✔ Results in '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()
