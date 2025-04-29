# snippet_bert.py

import pickle, numpy as np, html
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -- Load precomputed data --
_doc_ids, _doc_sents = pickle.load(open("doc_sentences.pkl", "rb"))
N, SENTS = len(_doc_sents), len(_doc_sents[0]) if _doc_sents else 0
# Memory‐map the quantized embeddings
# dtype=float16, shape=(N, SENTS, DIM)
# Replace DIM=384 if your model dims differ
DIM = 384  
_doc_embs = np.memmap("doc_sentence_embs.dat",
                     dtype=np.float16,
                     mode="r",
                     shape=(N, SENTS, DIM))

# A lightweight model just for query encoding
_query_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")


def get_best_snippet(query: str, doc_idx: int, max_length: int = 100) -> str:
    """
    Returns the single most query-relevant sentence for doc_idx.
    """
    sents = _doc_sents[doc_idx]
    embs16 = _doc_embs[doc_idx]          # float16 slice
    if embs16.size == 0 or not sents:
        return ""

    # Upcast and score
    embs32 = embs16.astype(np.float32)
    q_emb  = _query_model.encode([query], show_progress_bar=False)
    sims   = cosine_similarity(q_emb, embs32)[0]
    best   = int(np.argmax(sims))

    text = html.unescape(sents[best]).replace("\n", " ").strip()
    return (text[:max_length].rstrip() + "...") if len(text) > max_length else text
