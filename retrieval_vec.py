# retrieval_vec.py
from query_vector import clean_query
import numpy as np
import math
from bson_loader_1 import load_matrix
from metadata_loader import load_metadata
from query_vector import query_to_bow
from sklearn.preprocessing import normalize
from urllib.parse import urlparse

# BM25 parameters
k1, b = 1.2, 0.75
BLOCK_DOMAINS = {"odp.org", "amazon.com"}

# How many candidates to fetch before two‐stage rerank
CANDIDATE_POOL = 20

# Load once
print("🔄 Loading TF–IDF matrix for BM25 + cosine rerank…")
tfidf_matrix, doc_ids, vocab = load_matrix()   # sparse TF–IDF
meta = load_metadata()                          # now 5-tuples: (id, title, url, snippet, sents)
N, V = tfidf_matrix.shape

# Precompute IDF & TF matrices for BM25
csc = tfidf_matrix.tocsc()
df  = np.diff(csc.indptr)
idf = np.log((N - df + 0.5)/(df + 0.5) + 1)
tf_matrix = csc.multiply(1.0/idf).tocsr()

# Document lengths & average length
doc_lens = np.array(tf_matrix.sum(axis=1)).ravel()
avgdl    = doc_lens.mean()

# Precompute normalized TF–IDF for cosine stage
_mat_norm = normalize(tfidf_matrix, axis=1, copy=False)

# Build inverse vocab for title‐boost lookup
_inv_vocab = {i:t for t,i in vocab.items()}


def retrieve_bm25(query: str, top_k: int = 10):
    # Stage 0: preprocess query
    qbow = query_to_bow(query, vocab)
    if not qbow:
        return []

    # Stage 1: BM25 ranking
    bm25_scores = np.zeros(N, dtype=float)
    for term_idx, freq in qbow.items():
        w_idf    = idf[term_idx]
        tf_col   = tf_matrix[:, term_idx].toarray().ravel()
        denom    = tf_col + k1 * (1 - b + b * doc_lens / avgdl)
        bm25_scores += w_idf * ((k1 + 1) * tf_col) / denom

    # Top candidates
    cand_idxs = np.argsort(bm25_scores)[::-1][:CANDIDATE_POOL]

    # Stage 2: title‐boost & phrase bonus
    enhanced_scores = []
    phrase = query.lower()
    for idx in cand_idxs:
        # UNPACK 5-tuple (ignore sents list)
        doc_id, title, url, snippet, _ = meta[idx]
        domain = urlparse(url).netloc.lower()
        if any(bd in domain for bd in BLOCK_DOMAINS):
            enhanced_scores.append(-np.inf)
            continue

        score = bm25_scores[idx]
        title_lower = title.lower()
        for t_idx in qbow:
            term = _inv_vocab[t_idx]
            if term in title_lower:
                score += 0.1
        if phrase in snippet.lower():
            score += 0.5

        enhanced_scores.append(score)

    enhanced_scores = np.array(enhanced_scores)

    # Stage 3: cosine TF–IDF rerank
    qv = np.zeros(V, dtype=float)
    for term_idx, freq in qbow.items():
        qv[term_idx] = freq * idf[term_idx]
    norm = np.linalg.norm(qv)
    if norm > 0:
        qv /= norm

    cand_mat = _mat_norm[cand_idxs]
    cos_sims = cand_mat.dot(qv)

    final_scores = 0.5 * enhanced_scores + 0.5 * cos_sims

    # Pick final top_k
    top_hits = []
    seen    = set()
    top_inds = np.argsort(final_scores)[::-1]
    for i in top_inds:
        score_val = float(final_scores[i])
        # skip any non-finite scores
        if not math.isfinite(score_val):
            continue

        idx = cand_idxs[i]
        doc_id, title, url, snippet, _ = meta[idx]
        domain = urlparse(url).netloc.lower()
        if domain in seen:
            continue
        seen.add(domain)

        top_hits.append({
            "doc_id":  doc_id,
            "title":   title,
            "url":     url,
            "score":   score_val,
            "snippet": snippet
        })
        if len(top_hits) >= top_k:
            break

    return top_hits


def retrieve_vec(query: str, top_k: int = 10):
    return retrieve_bm25(query, top_k=top_k)
