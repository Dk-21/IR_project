# feedback_vec.py

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

import re
import numpy as np
from nltk.corpus import stopwords, wordnet

from bson_loader_1 import load_matrix
from retrieval_vec import retrieve_vec
from query_vector import clean_query, query_to_bow

# Rocchio parameters
TOP_K, NR_K          = 10, 10
ALPHA, BETA, GAMMA   = 1.0, 0.75, 0.15
EXP_TERMS            = 20

# Load TF–IDF matrix once
_mat, _doc_ids, _vocab = load_matrix()
_inv_vocab            = {i: t for t, i in _vocab.items()}
_id2idx               = {d: i for i, d in enumerate(_doc_ids)}

# English stopwords
STOP                  = set(stopwords.words("english"))


def _filter_terms(raw_terms):
    """
    Keep only:
     - lowercase alphabetic tokens
     - length ≥ 3
     - not NLTK stopwords
     - recognized by WordNet
    """
    out = []
    for t in raw_terms:
        if len(t) < 3:
            continue
        if not re.fullmatch(r"[a-z]+", t):
            continue
        if t in STOP:
            continue
        if not wordnet.synsets(t):  # only real English words
            continue
        out.append(t)
    return out


def rocchio_expand(query: str) -> str:
    # 1) Clean the query
    cq = clean_query(query)

    # 2) Retrieve relevant + nonrelevant
    hits     = retrieve_vec(cq, top_k=TOP_K + NR_K)
    rel_docs = hits[:TOP_K]
    nonrel   = hits[TOP_K : TOP_K + NR_K]

    # 3) Build original query vector
    qbow = query_to_bow(cq, _vocab)
    qv   = np.zeros(len(_vocab), dtype=float)
    for idx, freq in qbow.items():
        qv[idx] = freq

    # 4) Stack TF–IDF rows
    if rel_docs:
        rel_mat = np.vstack([
            _mat[_id2idx[d["doc_id"]]].toarray()[0]
            for d in rel_docs
        ])
    else:
        rel_mat = np.zeros((1, len(_vocab)))

    if nonrel:
        nr_mat = np.vstack([
            _mat[_id2idx[d["doc_id"]]].toarray()[0]
            for d in nonrel
        ])
    else:
        nr_mat = np.zeros_like(rel_mat)

    # 5) Centroids
    centroid_rel = rel_mat.mean(axis=0)
    centroid_nr  = nr_mat.mean(axis=0)

    # 6) Rocchio formula
    modified = ALPHA * qv + BETA * centroid_rel - GAMMA * centroid_nr

    # 7) Pick candidate indices (oversample)
    cand_idxs = np.argsort(modified)[-EXP_TERMS * 3 :]
    raw_terms = [ _inv_vocab[i] for i in cand_idxs if modified[i] > 0 ]

    # 8) Initial filter: stopwords, regex, WordNet
    filtered = _filter_terms(raw_terms)

    # 9) Require df≥2 in rel_docs & not in original
    orig_terms  = set(cq.split())
    final_terms = []
    for t in filtered:
        if t in orig_terms:
            continue
        tidx = _vocab[t]
        df_count = sum(_mat[_id2idx[d["doc_id"]], tidx] > 0 for d in rel_docs)
        if df_count >= 2:
            final_terms.append(t)

    # 10) Backfill to EXP_TERMS
    for t in filtered:
        if len(final_terms) >= EXP_TERMS:
            break
        if t not in final_terms and t not in orig_terms:
            final_terms.append(t)

    # 11) Build expanded query
    return query + " " + " ".join(final_terms)


def retrieve_expanded(query: str):
    exp_q = rocchio_expand(query)
    hits  = retrieve_vec(exp_q, top_k=TOP_K)
    return exp_q, hits
