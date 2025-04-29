# query_vector.py

import re
import nltk

# At the very top of your module—before any lemmatization calls:
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download once if you haven’t already:
# python -m nltk.downloader stopwords punkt

STOP = set(stopwords.words("english"))

def clean_query(text: str) -> str:
    """
    Lowercases, removes punctuation and numeric tokens,
    filters common stopwords, and returns the cleaned query string.
    """
    text = text.lower()
    # replace any non-letter with space
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    # keep only alphabetic tokens not in stoplist
    tokens = [t for t in tokens if t.isalpha() and t not in STOP]
    return " ".join(tokens)
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
