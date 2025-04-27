# query_vector.py

import re
import pickle
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import spacy
from bson_loader_1 import load_matrix
from metadata_loader import load_metadata

# load spaCy once
_nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

VOCAB_PATH = "vocab.pkl"
IDF_PATH   = "idf.npy"

def load_query_tools():
    # your existing corpus + vocab + idf loader
    mat, doc_ids, vocab = load_matrix()
    mat = normalize(mat, axis=1)
    meta = load_metadata()
    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    idf = np.load(IDF_PATH)
    return mat, doc_ids, meta, vocab, idf

def tokenize_and_lemmatize(text: str):
    """
    spaCy-based tokenizer + lemmatizer, filters non-alphanumerics.
    """
    doc = _nlp(text.lower())
    lemmas = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return lemmas

def query_to_vec(q: str) -> csr_matrix:
    """
    Build a normalized TF–IDF vector for the query string using lemmatized tokens.
    """
    # lemmas = tokenize_and_lemmatize(q)
    # counts = {}
    # for lemma in lemmas:
    #     if lemma in vocab:
    #         counts[lemma] = counts.get(lemma, 0) + 1

    # rows, cols, data = [], [], []
    # for term, tf in counts.items():
    #     col = vocab[term]
    #     rows.append(0)
    #     cols.append(col)
    #     data.append(tf * idf[col])
        # Load vocab & idf on‐the‐fly
    _, _, _, vocab, idf = load_query_tools()
    lemmas = tokenize_and_lemmatize(q)
    counts = {}
    for lemma in lemmas:
        if lemma in vocab:
            counts[lemma] = counts.get(lemma, 0) + 1

    rows, cols, data = [], [], []
    for term, tf in counts.items():
        col = vocab[term]
        rows.append(0)
        cols.append(col)
        data.append(tf * idf[col])

    qv = csr_matrix((data, (rows, cols)), shape=(1, idf.shape[0]))
    return normalize(qv, axis=1)

def retrieve_vec(query: str, top_k: int = 10):
    """
    Example retrieval function using the new query_to_vec.
    """
    from retrieval_vec import load_corpus_matrix  # or your existing loader
    mat, doc_ids, vocab, meta = load_query_tools()
    qv = query_to_vec(query, vocab, idf)
    sims = qv.dot(mat.T).toarray()[0]
    top_idxs = np.argsort(sims)[-top_k:][::-1]
    results = []
    for idx in top_idxs:
        results.append({
            "doc_id": doc_ids[idx],
            "title":  meta[idx][1],
            "url":    meta[idx][2],
            "score":  float(sims[idx])
        })
    return results
