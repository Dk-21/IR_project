# # feedback.py

# import numpy as np
# import pickle
# from bson_loader import load_corpus_matrix
# from retrieval import create_searcher, retrieve, RetrievalError

# # Rocchio parameters
# TOP_K           = 10
# NR_K            = 10
# ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
# EXPANSION_TERMS = 20

# # Load TF-IDF matrix and doc IDs
# tfidf_matrix, doc_ids = load_corpus_matrix()
# docid_to_idx = {doc_id:i for i,doc_id in enumerate(doc_ids)}

# # Load vectorizer for feature names
# with open("tfidf_vectorizer.pkl","rb") as vf:
#     vectorizer = pickle.load(vf)
# feature_names = vectorizer.get_feature_names_out()

# # Initialize Whoosh searcher/parser
# searcher, parser = create_searcher()

# def rocchio_expand(query_str: str) -> str:
#     try:
#         hits = retrieve(query_str, top_k=TOP_K+NR_K, searcher=searcher, parser=parser)
#     except RetrievalError as e:
#         raise RuntimeError(f"Initial retrieval failed: {e}")

#     rel_hits    = hits[:TOP_K]
#     nonrel_hits = hits[TOP_K:TOP_K+NR_K]

#     # Original query vector
#     q_vec = vectorizer.transform([query_str]).toarray()[0]

#     # Build centroids
#     def docs_to_matrix(hit_list):
#         idxs = [docid_to_idx[h["doc_id"]] for h in hit_list]
#         return tfidf_matrix[idxs].toarray()

#     R = docs_to_matrix(rel_hits).mean(axis=0)
#     N = docs_to_matrix(nonrel_hits).mean(axis=0)

#     q_mod = ALPHA*q_vec + BETA*R - GAMMA*N
#     topidx = np.argsort(q_mod)[-EXPANSION_TERMS:]
#     terms = feature_names[topidx]

#     # Filter out any OOV terms (just in case)
#     valid = set(feature_names)
#     clean = [t for t in terms if t in valid]

#     return query_str + " " + " ".join(clean)

# if __name__ == "__main__":
#     test = "marathon training plan for beginners"
#     print("Expanded:", rocchio_expand(test))
# feedback.py

import numpy as np
import pickle
from bson_loader import load_corpus_matrix
from retrieval import create_searcher, retrieve, RetrievalError
from whoosh.index import open_dir

# Rocchio parameters
TOP_K           = 10
NR_K            = 10
ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
EXPANSION_TERMS = 20
INDEX_DIR       = "whoosh_index"

# Load TF-IDF matrix and doc IDs
tfidf_matrix, doc_ids = load_corpus_matrix()
docid_to_idx = {doc_id: i for i, doc_id in enumerate(doc_ids)}

# Load vectorizer for feature names
with open("tfidf_vectorizer.pkl", "rb") as vf:
    vectorizer = pickle.load(vf)
feature_names = vectorizer.get_feature_names_out()

# Initialize Whoosh searcher/parser
searcher, parser = create_searcher()

# Prepare index reader for diagnostics
ix = open_dir(INDEX_DIR)
reader = ix.reader()


def rocchio_expand(query_str: str):
    """
    Generates an expanded query string via Rocchio feedback.
    """
    # Initial retrieval for feedback
    try:
        hits = retrieve(query_str, top_k=TOP_K + NR_K, searcher=searcher, parser=parser)
    except RetrievalError as e:
        raise RuntimeError(f"Initial retrieval failed: {e}")

    # Split relevant and non-relevant
    rel_hits = hits[:TOP_K]
    nonrel_hits = hits[TOP_K:TOP_K + NR_K]

    # Original query vector
    q_vec = vectorizer.transform([query_str]).toarray()[0]

    # Helper: docs -> matrix rows
    def hits_to_matrix(hit_list):
        idxs = [docid_to_idx[h['doc_id']] for h in hit_list]
        return tfidf_matrix[idxs].toarray()

    # Compute centroids
    centroid_rel = hits_to_matrix(rel_hits).mean(axis=0)
    centroid_nr = hits_to_matrix(nonrel_hits).mean(axis=0)

    # Rocchio formula
    q_mod = ALPHA * q_vec + BETA * centroid_rel - GAMMA * centroid_nr

    # Top candidate terms by weight
    top_idxs = np.argsort(q_mod)[-EXPANSION_TERMS:]
    candidates = feature_names[top_idxs]

    # Diagnostic: check term presence in index
    present_terms = [t for t in candidates if reader.has_word("content", t)]
    missing_terms = [t for t in candidates if t not in present_terms]
    if not present_terms:
        print(f"[Diagnostic] No expansion terms are present in the Whoosh index for query: {query_str!r}")
    else:
        print(f"[Diagnostic] Expansion terms present: {present_terms}")
    if missing_terms:
        print(f"[Diagnostic] Expansion terms missing: {missing_terms}")

    # Build cleaned expansion list
    clean_terms = present_terms
    # Use clean terms to form expanded query
    return query_str + (" " + " ".join(clean_terms) if clean_terms else "")


def rocchio_retrieve(query_str: str, top_k: int = TOP_K):
    """
    Expands query_str via Rocchio and retrieves top_k docs for the expanded query.
    Returns (expanded_query, [doc_ids]). If retrieval fails, returns empty list.
    """
    expanded_q = rocchio_expand(query_str)
    try:
        hits = retrieve(expanded_q, top_k=top_k, searcher=searcher, parser=parser)
        doc_ids = [h['doc_id'] for h in hits]
    except RetrievalError:
        doc_ids = []
        print(f"[Diagnostic] No documents retrieved for expanded query: {expanded_q!r}")
    return expanded_q, doc_ids

if __name__ == "__main__":
    test_q = "marathon training plan for beginners"
    print("Original:", test_q)
    exp_q = rocchio_expand(test_q)
    print("Expanded:", exp_q)
    exp_q2, exp_ids = rocchio_retrieve(test_q, top_k=10)
    print("Expanded IDs:", exp_ids)
