import re
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from idf_builder import load_idf


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def query_to_vec(query: str) -> csr_matrix:
    idf, vocab = load_idf()
    counts = {}
    for t in tokenize(query):
        if t in vocab:
            counts[t] = counts.get(t, 0) + 1
    rows, cols, data = [], [], []
    for term, tf in counts.items():
        col = vocab[term]
        rows.append(0)
        cols.append(col)
        data.append(tf * idf[col])
    vec = csr_matrix((data, (rows, cols)), shape=(1, len(idf)))
    return normalize(vec, axis=1)