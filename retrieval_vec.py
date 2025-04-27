# retrieval_vec.py

import numpy as np
from bson_loader_1 import load_matrix
from metadata_loader import load_metadata
from query_vector import query_to_vec
from sklearn.preprocessing import normalize
from urllib.parse import urlparse

# Now a list of blocked domains
BLOCK_DOMAINS = {"odp.org", "amazon.com"}

def retrieve_vec(query: str, top_k: int = 10):
    """
    Retrieves top_k documents by cosine similarity from the TF-IDF matrix,
    excludes any whose URL contains a blocked domain, and ensures at most one
    hit per (other) domain.
    """
    # 1) Load & normalize matrix
    mat, doc_ids, vocab = load_matrix()
    mat_norm = normalize(mat, axis=1)

    # 2) Build query vector
    qv = query_to_vec(query)

    # 3) Cosine similarities
    sims = qv.dot(mat_norm.T).toarray()[0]

    # 4) Sort all doc indices by descending score
    all_idxs = np.argsort(sims)[::-1]

    # 5) Load metadata for title/url lookup
    meta = load_metadata()

    # 6) Select top_k, filtering out blocked domains and duplicates
    results = []
    seen_domains = set()

    for idx in all_idxs:
        title, url = meta[idx][1], meta[idx][2]
        domain = urlparse(url).netloc.lower()

        # Skip if domain is in blocked list or we've already used it
        if any(block in domain for block in BLOCK_DOMAINS) or domain in seen_domains:
            continue

        results.append({
            "doc_id": doc_ids[idx],
            "title":  title,
            "url":    url,
            "score":  float(sims[idx])
        })
        seen_domains.add(domain)

        if len(results) >= top_k:
            break

    return results
