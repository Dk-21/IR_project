# query_vector.py

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter

# Simple tokenizer + stemmer + stop‐word removal
_stopset = set(stopwords.words("english"))
_stemmer = PorterStemmer()
_word_re = re.compile(r"\b[a-zA-Z]+\b")

def preprocess(text: str):
    tokens = _word_re.findall(text.lower())
    # drop stopwords, stem the rest
    return [_stemmer.stem(t) for t in tokens if t not in _stopset]

def query_to_bow(query: str, vocab: dict):
    """
    Turn a raw query into a dict {term_index: term_freq}
    using the same vocab (term→idx) as your TF–IDF matrix.
    """
    terms = preprocess(query)
    freqs = Counter(terms)
    bow = {}
    for t, f in freqs.items():
        if t in vocab:
            bow[vocab[t]] = f
    return bow
