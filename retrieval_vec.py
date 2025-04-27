# retrieval_vec.py

import numpy as np
from bson_loader_1 import load_matrix
from metadata_loader import load_metadata
from query_vector import query_to_vec
from sklearn.preprocessing import normalize


def retrieve_vec(query: str, top_k: int = 10):
    mat, doc_ids, vocab = load_matrix()
    mat_norm = normalize(mat, axis=1)
    qv = query_to_vec(query)
    # Compute cosine similarities
    sims = qv.dot(mat_norm.T).toarray()[0]
    # Select top-k indices
    idxs = np.argsort(sims)[-top_k:][::-1]
    meta = load_metadata()
    results = []
    for i in idxs:
        results.append({
            "doc_id": doc_ids[i],
            "title":  meta[i][1],
            "url":    meta[i][2],
            "score":  float(sims[i])
        })
    return results