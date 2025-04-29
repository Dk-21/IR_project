# precompute_snippets.py

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from metadata_loader import load_metadata

MODEL_NAME = "paraphrase-MiniLM-L3-v2"
meta       = load_metadata()   # list of (doc_id, title, url, snippet, [sents])

print("🔄 Loading BERT model for precompute…")
model = SentenceTransformer(MODEL_NAME)

doc_ids  = []
all_embs = []  # will store float16 arrays of shape (<=3, dim)

for doc_id, *_ , sents in meta:
    doc_ids.append(doc_id)
    if sents:
        embs32 = model.encode(sents, batch_size=16, show_progress_bar=False)
        embs16 = embs32.astype(np.float16)
    else:
        dim = model.get_sentence_embedding_dimension()
        embs16 = np.zeros((0, dim), dtype=np.float16)
    all_embs.append(embs16)

# 1) Save sentence lists for reference
with open("doc_sentences.pkl", "wb") as f:
    pickle.dump((doc_ids, [m[4] for m in meta]), f)

# 2) Build a memmap file: shape = (num_docs, MAX_SENTS, DIM)
N     = len(all_embs)
DIM   = all_embs[0].shape[1] if all_embs else 0
SENTS = max(arr.shape[0] for arr in all_embs)  # should be <=3

mp = np.memmap("doc_sentence_embs.dat", dtype=np.float16,
               mode="w+", shape=(N, SENTS, DIM))
for i, arr in enumerate(all_embs):
    mp[i, : arr.shape[0], :] = arr
mp.flush()

print(f"✔ Precomputed & memmapped embeddings for {N} docs → 'doc_sentence_embs.dat'")
