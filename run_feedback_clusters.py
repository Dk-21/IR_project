# # run_feedback_clusters.py

# import csv, numpy as np, pickle
# from retrieval import create_searcher, retrieve, RetrievalError
# from bson_loader import load_corpus_matrix
# from clusters import association_clusters, metric_clusters, scalar_clusters
# from feedback import TOP_K, NR_K, ALPHA, BETA, GAMMA, EXPANSION_TERMS

# SEED_FILE  = "50_queries.txt"
# OUTPUT_CSV = "clustered_feedback.csv"

# # Load TF-IDF matrix & doc IDs
# tfidf_matrix, doc_ids = load_corpus_matrix()
# with open("tfidf_vectorizer.pkl","rb") as f:
#     vectorizer = pickle.load(f)
# feature_names = vectorizer.get_feature_names_out()
# doc_to_idx   = {d:i for i,d in enumerate(doc_ids)}

# # Whoosh
# searcher, parser = create_searcher()

# def load_queries(path):
#     return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]

# def main():
#     queries = load_queries(SEED_FILE)
#     with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as fout:
#         w = csv.writer(fout)
#         w.writerow(["query","assoc_q","metric_q","scalar_q"])
#         for q in queries:
#             try:
#                 hits = retrieve(q, top_k=TOP_K+NR_K, searcher=searcher, parser=parser)
#             except RetrievalError:
#                 continue
#             rel, nonrel = hits[:TOP_K], hits[TOP_K:TOP_K+NR_K]
#             q_vec = vectorizer.transform([q]).toarray()[0]
#             R = tfidf_matrix[[doc_to_idx[h["doc_id"]] for h in rel]].toarray().mean(0)
#             N = tfidf_matrix[[doc_to_idx[h["doc_id"]] for h in nonrel]].toarray().mean(0)
#             q_mod = ALPHA*q_vec + BETA*R - GAMMA*N
#             topi = np.argsort(q_mod)[-EXPANSION_TERMS:]
#             candidates = feature_names[topi]

#             assoc  = association_clusters(list(candidates), [doc_to_idx[h["doc_id"]] for h in rel], tfidf_matrix, feature_names)
#             metric = metric_clusters(   list(candidates), [doc_to_idx[h["doc_id"]] for h in rel], tfidf_matrix, feature_names)
#             scalar = scalar_clusters(   list(candidates), vectorizer)

#             w.writerow([
#                 q,
#                 q + " " + " ".join(assoc),
#                 q + " " + " ".join(metric),
#                 q + " " + " ".join(scalar)
#             ])
#             print(f"Processed: {q!r}")

#     print(f"✔ Results in '{OUTPUT_CSV}'")

# if __name__ == "__main__":
#     main()
# run_feedback_clusters.py

# run_feedback_clusters.py

import csv
import pickle
import numpy as np
from bson_loader_1 import load_matrix
from metadata_loader import load_metadata
from retrieval_vec import retrieve_vec
from feeback_vec import rocchio_expand
from clusters import (
    get_top_terms,
    association_clusters,
    metric_clusters,
    scalar_clusters
)

# Config
SEED_FILE = "50_queries.txt"
TOP_K = 10
NUM_CLUSTERS = 5
RESULTS_CSV = "feedback_clusters_results.csv"

# Load TF-IDF matrix, vocab, and metadata once
tfidf_matrix, doc_ids, vocab = load_matrix()
meta = load_metadata()  # list of (doc_id, title, url)
# Map doc_id -> index
docid_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
# Build feature_names array
element_count = len(vocab)
feature_names = np.empty(element_count, dtype=object)
for term, idx in vocab.items():
    feature_names[idx] = term

# Load seed queries
with open(SEED_FILE, encoding="utf-8") as f:
    queries = [line.strip() for line in f if line.strip()]

# Prepare CSV writer
with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    writer.writerow([
        "query", "method", "expanded_terms", "rank", "title", "url", "score"
    ])

    for query in queries:
        print(f"Processing: '{query}'")

        # 1) Baseline TF-IDF retrieval
        base_hits = retrieve_vec(query, top_k=TOP_K)
        rel_idxs = [docid_to_idx[h['doc_id']] for h in base_hits]
        for rank, h in enumerate(base_hits, start=1):
            writer.writerow([query, "baseline", "", rank, h['title'], h['url'], f"{h['score']:.6f}"])

        # 2) Rocchio expansion
        exp_q = rocchio_expand(query)
        exp_terms = exp_q.replace(query, '').strip().split()
        rocchio_hits = retrieve_vec(exp_q, top_k=TOP_K)
        for rank, h in enumerate(rocchio_hits, start=1):
            writer.writerow([query, "rocchio", ";".join(exp_terms), rank, h['title'], h['url'], f"{h['score']:.6f}"])

        # Compute candidates from top TF-IDF terms in relevant docs
        candidates = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=50)

        # 3) Association clusters
        assoc_terms = association_clusters(candidates, rel_idxs, tfidf_matrix, feature_names)
        assoc_q = query + " " + " ".join(assoc_terms)
        assoc_hits = retrieve_vec(assoc_q, top_k=TOP_K)
        for rank, h in enumerate(assoc_hits, start=1):
            writer.writerow([query, "association", ";".join(assoc_terms), rank, h['title'], h['url'], f"{h['score']:.6f}"])

        # 4) Metric clusters
        metric_terms = metric_clusters(candidates, rel_idxs, tfidf_matrix, feature_names, n_clusters=NUM_CLUSTERS)
        metric_q = query + " " + " ".join(metric_terms)
        metric_hits = retrieve_vec(metric_q, top_k=TOP_K)
        for rank, h in enumerate(metric_hits, start=1):
            writer.writerow([query, "metric", ";".join(metric_terms), rank, h['title'], h['url'], f"{h['score']:.6f}"])

        # 5) Scalar clusters
        scalar_terms = scalar_clusters(candidates, rel_idxs, tfidf_matrix, feature_names, n_buckets=NUM_CLUSTERS)
        scalar_q = query + " " + " ".join(scalar_terms)
        scalar_hits = retrieve_vec(scalar_q, top_k=TOP_K)
        for rank, h in enumerate(scalar_hits, start=1):
            writer.writerow([query, "scalar", ";".join(scalar_terms), rank, h['title'], h['url'], f"{h['score']:.6f}"])

print(f"✔ Results written to '{RESULTS_CSV}'")
