# clusters.py

import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix


def get_top_terms(
    rel_idxs: list,
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    top_n: int = 50
) -> list:
    """
    Returns the top_n terms by summed TF-IDF score across the relevant documents.
    """
    # Subset the TF-IDF matrix to relevant documents
    submat = tfidf_matrix[rel_idxs]
    # Sum TF-IDF scores for each term (column)
    term_scores = np.array(submat.sum(axis=0)).ravel()
    # Get indices of the top_n scoring terms
    top_indices = np.argsort(term_scores)[-top_n:]
    # Map back to feature names
    return [feature_names[i] for i in top_indices]


def association_clusters(
    candidates: list,
    rel_idxs: list,
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    threshold: int = 1
) -> list:
    """
    Clusters candidate terms by co-occurrence in the relevant docs,
    then picks one representative term (highest avg TF-IDF) per cluster.
    """
    # Map term names to column indices
    name_to_idx = {t: i for i, t in enumerate(feature_names)}
    # Filter out any candidates not in the vocabulary
    filtered = [t for t in candidates if t in name_to_idx]
    cidx = [name_to_idx[t] for t in filtered]
    if not cidx:
        return []

    # Build a binary occurrence matrix: docs x terms
    occ = (tfidf_matrix[rel_idxs][:, cidx].toarray() > 0)
    n = len(cidx)

    # Build adjacency list based on co-occurrence threshold
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if np.logical_and(occ[:, i], occ[:, j]).sum() >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # Find connected components
    seen = set()
    clusters = []
    for i in range(n):
        if i not in seen:
            stack = [i]
            comp = set()
            while stack:
                u = stack.pop()
                if u not in seen:
                    seen.add(u)
                    comp.add(u)
                    stack.extend(adj[u] - seen)
            clusters.append(comp)

    # From each cluster, pick the term with highest average TF-IDF
    weights = tfidf_matrix[rel_idxs][:, cidx].toarray()
    selected = []
    for comp in clusters:
        best = max(comp, key=lambda idx: weights[:, idx].mean())
        selected.append(filtered[best])
    return selected


def metric_clusters(
    candidates: list,
    rel_idxs: list,
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    n_clusters: int = 4
) -> list:
    """
    Uses KMeans to cluster candidate term vectors (across relevant docs),
    then picks one representative term from each cluster.
    """
    # Term name -> index mapping
    name_to_idx = {t: i for i, t in enumerate(feature_names)}
    filtered = [t for t in candidates if t in name_to_idx]
    cidx = [name_to_idx[t] for t in filtered]
    if not cidx:
        return []

    # Prepare term-document matrix for clustering: terms x docs
    X = tfidf_matrix[rel_idxs][:, cidx].toarray().T
    k = min(n_clusters, len(filtered))
    if k <= 0:
        return []

    # Perform KMeans clustering
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

    # Pick the highest-average TF-IDF term per cluster
    weights = tfidf_matrix[rel_idxs][:, cidx].toarray()
    selected = []
    for lbl in sorted(set(labels)):
        members = [i for i, lab in enumerate(labels) if lab == lbl]
        best = max(members, key=lambda idx: weights[:, idx].mean())
        selected.append(filtered[best])
    return selected


def scalar_clusters(
    candidates: list,
    rel_idxs: list,
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    n_buckets: int = 3
) -> list:
    """
    Buckets candidates by global IDF (computed from the entire corpus),
    then selects the highest-IDF term within each bucket.
    """
    # Compute global DF and IDF for all terms
    N = tfidf_matrix.shape[0]
    csc = tfidf_matrix.tocsc()
    df = np.diff(csc.indptr)
    idf = np.log((N + 1) / (df + 1)) + 1

    # Map names to indices and collect IDF for candidates
    name_to_idx = {t: i for i, t in enumerate(feature_names)}
    term_idfs = [(t, idf[name_to_idx[t]]) for t in candidates if t in name_to_idx]
    if not term_idfs:
        return []

    # Sort by IDF and split into buckets
    term_idfs.sort(key=lambda x: x[1])
    buckets = np.array_split(term_idfs, n_buckets)

    # Pick top IDF term in each bucket
    selected = []
    for bucket in buckets:
        if len(bucket) > 0:
            term, _ = max(bucket, key=lambda x: x[1])
            selected.append(term)
    return selected
