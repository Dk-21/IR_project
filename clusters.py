# clusters.py

import re
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
import nltk

# At the very top of your module—before any lemmatization calls:
nltk.download("wordnet", quiet=True)
# If you also need the “omw-1.4” data for some language mappings:
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ——— Configuration ———
MIN_TERM_LEN = 4           # drop very short terms
STOPWORDS    = set(stopwords.words("english"))
LEMMA        = WordNetLemmatizer()
VALID_TOKEN  = re.compile(r"^[a-zA-Z]+$")  # only letters

def _clean_terms(candidates):
    """
    Lowercase, alphabetic-only, length filter, stopword removal,
    and lemmatization of candidates.
    """
    cleaned = []
    for t in candidates:
        t0 = t.lower().strip()
        if not VALID_TOKEN.match(t0):
            continue
        if len(t0) < MIN_TERM_LEN or t0 in STOPWORDS:
            continue
        cleaned.append(LEMMA.lemmatize(t0))
    # de-dupe while preserving order
    seen = set()
    out  = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out



import re, numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from nltk.corpus import stopwords, wordnet
import spacy

# ——— CONFIG ———
MIN_LEN      = 4
STOPWORDS    = set(stopwords.words("english"))
CODE_BLACK   = {"var","function","return","false","true","comment","reply",
                "http","html","div","span","class","id","src"}
_nlp         = spacy.load("en_core_web_sm", disable=["parser","ner"])
WORD_RE      = re.compile(r"^[a-zA-Z]+$")


def is_valid_term(t: str) -> bool:
    t0 = t.lower()
    if t0 in STOPWORDS or t0 in CODE_BLACK: 
        return False
    if len(t0) < MIN_LEN or not WORD_RE.match(t0):
        return False
    if not wordnet.synsets(t0):
        return False
    return True


def pos_filter(tokens: list[str]) -> list[str]:
    doc = _nlp(" ".join(tokens))
    return [tok.text for tok in doc if tok.pos_ in {"NOUN","VERB","ADJ"}]


def get_top_terms(
    rel_idxs: list[int],
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    top_n: int = 50
) -> list[str]:
    # 1) Sum TF-IDF across rel_docs
    term_scores = np.array(tfidf_matrix[rel_idxs].sum(axis=0)).ravel()
    top_idxs    = np.argsort(term_scores)[-top_n:]
    raw_terms   = [feature_names[i] for i in top_idxs]
    raw_scores  = [term_scores[i] for i in top_idxs]

    # 2) Threshold: only keep terms ≥ median of these top_n
    med = np.median(raw_scores)
    filtered = [t.lower() for t, s in zip(raw_terms, raw_scores) if s >= med]

    # 3) POS‐filter & WordNet & length/stop/code blacklist
    posed = pos_filter(filtered)
    final = [t for t in posed if is_valid_term(t)]

    # 4) Dedupe & preserve order
    seen, out = set(), []
    for t in final:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def association_clusters(
    candidates: list,
    rel_idxs: list,
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    threshold: int = 2
) -> list:
    """
    Co‐occurrence clustering: pick one term per connected component.
    """
    cands = _clean_terms(candidates)
    if not cands:
        return []

    idx_map = {t:i for i,t in enumerate(feature_names)}
    # filter out any candidate not in vocab
    valid   = [t for t in cands if t in idx_map]
    cidx    = [idx_map[t] for t in valid]

    occ     = (tfidf_matrix[rel_idxs][:, cidx].toarray() > 0)
    n       = len(cidx)

    # build adjacency
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            if np.logical_and(occ[:,i], occ[:,j]).sum() >= threshold:
                adj[i].add(j)
                adj[j].add(i)

    # find connected components
    seen, clusters = set(), []
    for i in range(n):
        if i in seen: continue
        stack, comp = [i], set()
        while stack:
            u = stack.pop()
            if u not in seen:
                seen.add(u)
                comp.add(u)
                stack.extend(adj[u] - seen)
        clusters.append(comp)

    # pick the highest‐avg TF–IDF term per cluster
    weights = tfidf_matrix[rel_idxs][:, cidx].toarray()
    selected = []
    for comp in clusters:
        best = max(comp, key=lambda i: weights[:,i].mean())
        selected.append(valid[best])
    return selected


def metric_clusters(
    candidates: list,
    rel_idxs: list,
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    n_clusters: int = 4
) -> list:
    """
    K‐means clustering over term vectors (terms × docs).
    """
    cands = _clean_terms(candidates)
    if not cands:
        return []

    idx_map = {t:i for i,t in enumerate(feature_names)}
    valid   = [t for t in cands if t in idx_map]
    cidx    = [idx_map[t] for t in valid]
    if not cidx:
        return []

    # term-document matrix: terms × docs
    X      = tfidf_matrix[rel_idxs][:, cidx].toarray().T
    k      = min(n_clusters, len(valid))
    labels = KMeans(n_clusters=k, random_state=42).fit_predict(X)

    # pick highest‐avg TF–IDF term per cluster
    weights = tfidf_matrix[rel_idxs][:, cidx].toarray()
    selected = []
    for lbl in sorted(set(labels)):
        members = [i for i,l in enumerate(labels) if l==lbl]
        best    = max(members, key=lambda i: weights[:,i].mean())
        selected.append(valid[best])
    return selected


def scalar_clusters(
    candidates: list,
    rel_idxs: list,
    tfidf_matrix: csr_matrix,
    feature_names: np.ndarray,
    n_buckets: int = 3
) -> list:
    """
    Buckets candidates by global IDF, then takes the top‐IDF per bucket.
    """
    cands = _clean_terms(candidates)
    if not cands:
        return []

    # compute global IDF
    N   = tfidf_matrix.shape[0]
    csc = tfidf_matrix.tocsc()
    df  = np.diff(csc.indptr)
    idf = np.log((N + 1)/(df + 1)) + 1

    idx_map = {t:i for i,t in enumerate(feature_names)}
    term_idfs = [(t, idf[idx_map[t]]) for t in cands if t in idx_map]
    if not term_idfs:
        return []

    term_idfs.sort(key=lambda x: x[1])
    buckets = np.array_split(term_idfs, n_buckets)

    selected = []
    for bucket in buckets:
        if len(bucket) > 0:
            term, _ = max(bucket, key=lambda x: x[1])
            selected.append(term)
    return selected
