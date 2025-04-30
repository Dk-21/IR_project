# build_running_vocab.py

import os
import re
import numpy as np
import joblib
import scipy.sparse

from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec

# Adjust these imports to wherever you keep your loaders:
from bson_loader_1    import load_matrix_direct
from metadata_loader  import load_metadata

# ——— Parameters ———
NUM_SEED_DOCS    = 1000    # how many "run" docs to sample
TOP_TFIDF_TERMS  = 500     # harvest this many top TF–IDF terms
EMBED_NEIGHBORS  = 10      # neighbors per seed in Word2Vec
DF_THRESHOLD     = 5       # min doc‐freq for pruning
VOCAB_OUT        = "running_vocab.txt"

# 1) Load TF–IDF matrix & vocab
print("Loading TF–IDF matrix …")
mat, doc_ids, vocab = load_matrix_direct()

# 2) Load metadata; each entry is (doc_id, title, url, snippet, sents)
print("Loading metadata …")
meta = load_metadata()

# 3) Select seed docs whose URL contains "run"
run_idxs = []
for i, entry in enumerate(meta):
    url = entry[2]  # third element is URL
    if url and "run" in url.lower():
        run_idxs.append(i)
    if len(run_idxs) >= NUM_SEED_DOCS:
        break

# 4) Sum TF–IDF over those docs → pick top terms
submat     = mat[run_idxs]                  # shape (NUM_SEED_DOCS × V)
term_sums  = np.array(submat.sum(axis=0)).ravel()
top_idxs   = np.argsort(term_sums)[-TOP_TFIDF_TERMS:]
inv_vocab  = {i: t for t, i in vocab.items()}
tfidf_terms = [inv_vocab[i] for i in top_idxs]

# 5) WordNet expansions
wn_terms = set()
for term in tfidf_terms:
    for syn in wn.synsets(term):
        for lemma in syn.lemmas():
            w = lemma.name().lower().replace("_", "")
            if w.isalpha() and w not in tfidf_terms:
                wn_terms.add(w)

# 6) Train Word2Vec on your snippets (or full text if you prefer)
print("Training Word2Vec on snippets …")
sentences = []
for entry in meta:
    snippet = entry[3] or ""
    tokens  = re.findall(r"[a-zA-Z]+", snippet.lower())
    sentences.append(tokens)

w2v = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=5,
    workers=os.cpu_count()
)

wv_terms = set()
for term in tfidf_terms:
    if term in w2v.wv:
        for neigh, _ in w2v.wv.most_similar(term, topn=EMBED_NEIGHBORS):
            if neigh.isalpha() and neigh not in tfidf_terms:
                wv_terms.add(neigh)

# 7) Combine all candidates and prune by DF
all_cands = set(tfidf_terms) | wn_terms | wv_terms
pruned    = []
for t in all_cands:
    idx = vocab.get(t)
    if idx is None:
        continue
    df = int((mat[:, idx] > 0).sum())
    if df >= DF_THRESHOLD:
        pruned.append(t)

# 8) Write out the final vocab
with open(VOCAB_OUT, "w", encoding="utf-8") as f:
    for term in sorted(pruned):
        f.write(term + "\n")

print(f"✔ Wrote {len(pruned)} terms to '{VOCAB_OUT}'")
