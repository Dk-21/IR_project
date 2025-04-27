# feedback_vec.py

import numpy as np
from bson_loader_1 import load_matrix
from retrieval_vec import retrieve_vec
from query_vector import query_to_vec

# Rocchio parameters
TOP_K, NR_K = 10, 10
ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
EXP_TERMS = 20


def rocchio_expand(query: str) -> str:
    hits = retrieve_vec(query, top_k=TOP_K + NR_K)
    rel    = hits[:TOP_K]
    nonrel = hits[TOP_K:TOP_K + NR_K]

    mat, doc_ids, vocab = load_matrix()
    id2idx = {d:i for i,d in enumerate(doc_ids)}

    qv = query_to_vec(query).toarray()[0]
    rel_mat = np.array([mat[id2idx[h["doc_id"]]].toarray()[0] for h in rel])
    nr_mat  = np.array([mat[id2idx[h["doc_id"]]].toarray()[0] for h in nonrel]) if nonrel else np.zeros_like(rel_mat)

    centroid_rel = rel_mat.mean(axis=0)
    centroid_nr  = nr_mat.mean(axis=0)

    q_mod = ALPHA*qv + BETA*centroid_rel - GAMMA*centroid_nr
    topi = np.argsort(q_mod)[-EXP_TERMS:]
    inv_vocab = {v:k for k,v in vocab.items()}
    terms = [inv_vocab[i] for i in topi]
    return query + " " + " ".join(terms)


def retrieve_expanded(query: str):
    exp_q = rocchio_expand(query)
    hits = retrieve_vec(exp_q, top_k=TOP_K)
    return exp_q, hits