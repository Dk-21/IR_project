# clusters.py

import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from typing import List

def association_clusters(
    candidates: List[str],
    rel_idxs: List[int],
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    threshold: int = 1
) -> List[str]:
    name_to_idx = {t:i for i,t in enumerate(feature_names)}
    cidx = [name_to_idx[t] for t in candidates]
    binmat = (tfidf_matrix[rel_idxs][:,cidx].toarray() > 0)
    n = len(cidx)

    # Build adjacency
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            if np.logical_and(binmat[:,i], binmat[:,j]).sum() >= threshold:
                adj[i].add(j); adj[j].add(i)

    # Connected components
    seen, clusters = set(), []
    for i in range(n):
        if i not in seen:
            stack, comp = [i], set()
            while stack:
                u = stack.pop()
                if u not in seen:
                    seen.add(u); comp.add(u)
                    stack.extend(adj[u] - seen)
            clusters.append(comp)

    # Pick highest-average TF-IDF term per cluster
    weights = tfidf_matrix[rel_idxs][:,cidx].toarray()
    selected = []
    for comp in clusters:
        best = max(comp, key=lambda i: weights[:,i].mean())
        selected.append(candidates[best])
    return selected

def metric_clusters(
    candidates: List[str],
    rel_idxs: List[int],
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    n_clusters: int = 4
) -> List[str]:
    name_to_idx = {t:i for i,t in enumerate(feature_names)}
    cidx = [name_to_idx[t] for t in candidates]
    X = tfidf_matrix[rel_idxs][:,cidx].toarray().T
    k = min(n_clusters, len(candidates))
    if k < 1:
        return []
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)
    weights = tfidf_matrix[rel_idxs][:,cidx].toarray()

    selected = []
    for lbl in sorted(set(labels)):
        members = [i for i,l in enumerate(labels) if l==lbl]
        best    = max(members, key=lambda i: weights[:,i].mean())
        selected.append(candidates[best])
    return selected

def scalar_clusters(
    candidates: List[str],
    vectorizer,
    n_buckets: int = 3
) -> List[str]:
    idf = vectorizer.idf_
    fn  = vectorizer.get_feature_names_out()
    name_to_idx = {t:i for i,t in enumerate(fn)}

    term_idfs = [(t, idf[name_to_idx[t]]) for t in candidates if t in name_to_idx]
    term_idfs.sort(key=lambda x: x[1])
    buckets = np.array_split(term_idfs, n_buckets)

    selected = []
    for bucket in buckets:
        if bucket.size:
            term, _ = max(bucket, key=lambda x: x[1])
            selected.append(term)
    return selected
