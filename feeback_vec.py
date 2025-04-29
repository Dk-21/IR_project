# # feedback_vec.py

# import numpy as np
# from bson_loader_1 import load_matrix
# from retrieval_vec import retrieve_vec
# from query_vector import query_to_vec

# # Rocchio parameters
# TOP_K, NR_K = 10, 10
# ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
# EXP_TERMS = 20


# def rocchio_expand(query: str) -> str:
#     hits = retrieve_vec(query, top_k=TOP_K + NR_K)
#     rel    = hits[:TOP_K]
#     nonrel = hits[TOP_K:TOP_K + NR_K]

#     mat, doc_ids, vocab = load_matrix()
#     id2idx = {d:i for i,d in enumerate(doc_ids)}

#     qv = query_to_vec(query).toarray()[0]
#     rel_mat = np.array([mat[id2idx[h["doc_id"]]].toarray()[0] for h in rel])
#     nr_mat  = np.array([mat[id2idx[h["doc_id"]]].toarray()[0] for h in nonrel]) if nonrel else np.zeros_like(rel_mat)

#     centroid_rel = rel_mat.mean(axis=0)
#     centroid_nr  = nr_mat.mean(axis=0)

#     q_mod = ALPHA*qv + BETA*centroid_rel - GAMMA*centroid_nr
#     topi = np.argsort(q_mod)[-EXP_TERMS:]
#     inv_vocab = {v:k for k,v in vocab.items()}
#     terms = [inv_vocab[i] for i in topi]
#     return query + " " + " ".join(terms)


# def retrieve_expanded(query: str):
#     exp_q = rocchio_expand(query)
#     hits = retrieve_vec(exp_q, top_k=TOP_K)
#     return exp_q, hits
# feedback_vec.py

import numpy as np
from bson_loader_1 import load_matrix
from retrieval_vec import retrieve_vec, vocab, doc_ids
from query_vector import query_to_bow

# Rocchio parameters
TOP_K, NR_K   = 10, 10
ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
EXP_TERMS     = 20

# --- Preload TF–IDF matrix once ---
# mat: sparse TF–IDF matrix; doc_ids: list of IDs; vocab: term→index
_mat, _doc_ids, _vocab = load_matrix()
# Build inverse vocab for term lookup
_inv_vocab = {i:t for t,i in _vocab.items()}
# Map doc_id → row index for quick lookup
_id2idx = {d:i for i,d in enumerate(_doc_ids)}


def rocchio_expand(query: str) -> str:
    """
    Returns the original query string PLUS the top EXP_TERMS Rocchio expansion terms.
    """
    # 1) Retrieve baseline and non‐relevant docs
    hits = retrieve_vec(query, top_k=TOP_K + NR_K)
    rel_docs    = hits[:TOP_K]
    nonrel_docs = hits[TOP_K:TOP_K + NR_K]

    # 2) Build the original query vector in TF–IDF space
    #    query_to_bow returns {term_idx: freq}
    q_bow = query_to_bow(query, _vocab)
    qv = np.zeros(len(_vocab), dtype=float)
    for idx, freq in q_bow.items():
        # you could weight by idf here if you want; for now raw tf
        qv[idx] = freq

    # 3) Stack TF–IDF vectors of rel / nonrel docs
    rel_mat = np.vstack([
        _mat[_id2idx[d["doc_id"]]].toarray()[0]
        for d in rel_docs
    ]) if rel_docs else np.zeros((1, len(_vocab)))

    nr_mat = np.vstack([
        _mat[_id2idx[d["doc_id"]]].toarray()[0]
        for d in nonrel_docs
    ]) if nonrel_docs else np.zeros_like(rel_mat)

    # 4) Compute centroids
    centroid_rel = rel_mat.mean(axis=0)
    centroid_nr  = nr_mat.mean(axis=0)

    # 5) Rocchio formula
    modified = ALPHA * qv + BETA * centroid_rel - GAMMA * centroid_nr

    # 6) Pick top EXP_TERMS highest‐weight terms (excluding zero or original query terms)
    top_indices = np.argsort(modified)[-EXP_TERMS:]
    # filter out any term with non‐positive weight
    top_indices = [i for i in top_indices if modified[i] > 0]

    expansion_terms = [_inv_vocab[i] for i in top_indices]
    # 7) Return the expanded query string
    return query + " " + " ".join(expansion_terms)


def retrieve_expanded(query: str):
    """
    Expands `query` via Rocchio, then runs the same `retrieve_vec` on it.
    Returns (expanded_query_string, hits_list).
    """
    expanded_q = rocchio_expand(query)
    expanded_hits = retrieve_vec(expanded_q, top_k=TOP_K)
    return expanded_q, expanded_hits
