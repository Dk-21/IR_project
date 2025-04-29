# retrieval_vec.py

import numpy as np
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
meta = load_metadata()                          # [(doc_id, title, url, snippet), …]
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
    """
    1) BM25 on the entire corpus → top CANDIDATE_POOL candidates
    2) Title‐boost & phrase bonus
    3) Two‐stage cosine TF–IDF rerank → final top_k
    """
    # --- Stage 0: preprocess query ---
    qbow = query_to_bow(query, vocab)    # {term_idx: freq}
    if not qbow:
        return []

    # --- Stage 1: BM25 ranking ---
    bm25_scores = np.zeros(N, dtype=float)
    for term_idx, qf in qbow.items():
        w_idf = idf[term_idx]
        tf_col = tf_matrix[:, term_idx].toarray().ravel()
        denom = tf_col + k1 * (1 - b + b * doc_lens / avgdl)
        bm25_scores += w_idf * ((k1+1) * tf_col) / denom

    # Grab top‐CANDIDATE_POOL indices
    cand_idxs = np.argsort(bm25_scores)[::-1][:CANDIDATE_POOL]

    # --- Stage 2: Apply title‐boost & phrase bonus ---
    enhanced_scores = []
    phrase = query.lower()
    for idx in cand_idxs:
        doc_id, title, url, snippet = meta[idx]
        domain = urlparse(url).netloc.lower()
        if any(b in domain for b in BLOCK_DOMAINS):
            enhanced_scores.append(-np.inf)
            continue

        score = bm25_scores[idx]

        # title boost: +0.1 per query term in title
        title_lower = title.lower()
        for term_idx in qbow.keys():
            term = _inv_vocab[term_idx]
            if term in title_lower:
                score += 0.1

        # phrase bonus: +0.5 if full query appears in snippet
        if phrase in snippet.lower():
            score += 0.5

        enhanced_scores.append(score)

    enhanced_scores = np.array(enhanced_scores)

    # --- Stage 3: Cosine TF–IDF rerank of the candidate pool ---
    # Build query TF–IDF vector
    qv = np.zeros(V, dtype=float)
    for term_idx, freq in qbow.items():
        qv[term_idx] = freq * idf[term_idx]
    norm = np.linalg.norm(qv)
    if norm > 0:
        qv /= norm

    # Cosine sims against normalized matrix rows
    cand_mat = _mat_norm[cand_idxs]
    cos_sims = cand_mat.dot(qv)

    # Combine the two scores (e.g. 50% BM25‐enhanced, 50% cosine)
    final_scores = 0.5 * enhanced_scores + 0.5 * cos_sims

    # --- Pick top_k from the candidate pool ---
    top_inds = np.argsort(final_scores)[::-1][:top_k]
    hits = []
    seen = set()
    for i in top_inds:
        idx = cand_idxs[i]
        doc_id, title, url, snippet = meta[idx]
        domain = urlparse(url).netloc.lower()
        if domain in seen:
            continue
        seen.add(domain)
        hits.append({
            "doc_id":  doc_id,
            "title":   title,
            "url":     url,
            "score":   float(final_scores[i]),
            "snippet": snippet
        })
        if len(hits) >= top_k:
            break

    return hits

# Alias for your existing API
def retrieve_vec(query: str, top_k: int = 10):
    return retrieve_bm25(query, top_k=top_k)
