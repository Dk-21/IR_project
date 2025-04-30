# feedback_vec.py

import re
import numpy as np
import nltk
import os
# Ensure NLTK data is available
nltk.download("stopwords", quiet=True)
VOCAB_FILE = os.path.join(os.path.dirname(__file__), "running_vocab.txt")
with open(VOCAB_FILE, encoding="utf-8") as f:
    RUNNING_VOCAB = [line.strip() for line in f if line.strip()]
from nltk.corpus import stopwords
from bson_loader_1 import load_matrix
from retrieval_vec   import retrieve_vec
from query_vector    import clean_query, query_to_bow

# ——— Rocchio parameters ———
TOP_K, NR_K        = 10, 10
ALPHA, BETA, GAMMA = 1.0, 0.75, 0.15
EXP_TERMS          = 10   # how many expansion terms you want

# ——— Load TF–IDF matrix once ———
_mat, _doc_ids, _vocab = load_matrix()
_inv_vocab             = {i: t for t, i in _vocab.items()}
_id2idx                = {d: i for i, d in enumerate(_doc_ids)}

# ——— A small curated running‐domain vocabulary ———
RUNNING_VOCAB = [
    "marathon", "training", "run", "recovery", "nutrition",
    "pace", "endurance", "cadence", "tempo", "hydration",
    "injury", "shoe", "interval", "trail", "crosscountry"
]

# Optional basic filter (remove non‐alpha, stopwords)
WORD_RE   = re.compile(r"^[a-z]+$")
STOPWORDS = set(stopwords.words("english"))


def _clean_term(t: str) -> str:
    t2 = t.lower()
    if not WORD_RE.fullmatch(t2):
        return ""
    if t2 in STOPWORDS:
        return ""
    return t2

def rocchio_expand(query: str) -> str:
    """
    1) Clean query
    2) Retrieve TOP_K+NR_K docs
    3) Build modified vector
    4) Pick positive terms
    5) Basic cleaning
    6) Keep only those in RUNNING_VOCAB
    7) Pad to EXP_TERMS
    8) Return expanded query
    """
    # 1) Clean
    cq = clean_query(query)

    # 2) Retrieve
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
            _mat[_id2idx[h["doc_id"]]].toarray()[0]
            for h in rel_docs
        ])
    else:
        rel_mat = np.zeros((1, len(_vocab)))

    if nonrel:
        nr_mat = np.vstack([
            _mat[_id2idx[h["doc_id"]]].toarray()[0]
            for h in nonrel
        ])
    else:
        nr_mat = np.zeros_like(rel_mat)

    # 5) Rocchio formula
    centroid_rel = rel_mat.mean(axis=0)
    centroid_nr  = nr_mat.mean(axis=0)
    modified     = ALPHA * qv + BETA * centroid_rel - GAMMA * centroid_nr

    # 6) Pick positive-weight indices, sort descending, take 3×EXP_TERMS
    pos_idxs  = np.where(modified > 0)[0]
    # sort by weight
    sorted_pos = pos_idxs[np.argsort(modified[pos_idxs])[::-1]]
    cand_idxs  = sorted_pos[: EXP_TERMS * 3]
    raw_terms  = [ _inv_vocab[i] for i in cand_idxs ]

    # 7) Basic cleaning + dedup
    seen, cleaned = set(), []
    for t in raw_terms:
        ct = _clean_term(t)
        if not ct or ct in seen:
            continue
        seen.add(ct)
        cleaned.append(ct)
        if len(cleaned) >= EXP_TERMS * 3:
            break

    # 8) Keep only those in your running vocab
    domain_terms = [t for t in cleaned if t in RUNNING_VOCAB]

    # 9) Pad out to exactly EXP_TERMS
    final_terms = []
    for t in domain_terms:
        if len(final_terms) >= EXP_TERMS:
            break
        final_terms.append(t)

    for seed in RUNNING_VOCAB:
        if len(final_terms) >= EXP_TERMS:
            break
        if seed not in final_terms and seed not in cq.split():
            final_terms.append(seed)

    # 10) Build and return
    if final_terms:
        return query + " " + " ".join(final_terms)
    else:
        return query  # no expansions


def retrieve_expanded(query: str):
    """
    Returns (expanded_query, hits) for use in your API.
    """
    exp_q = rocchio_expand(query)
    hits  = retrieve_vec(exp_q, top_k=TOP_K)
    return exp_q, hits
