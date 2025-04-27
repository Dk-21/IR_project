# semantic_retrieval.py

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from bson import decode_file_iter
from metadata_loader import load_metadata

# Model for encoding queries and docs
MODEL_NAME = 'all-MiniLM-L6-v2'
BSON_PATH = 'web_crawl/pages.bson'
EMBED_PATH = 'doc_embeddings.npy'
DOCIDS_PATH = 'doc_ids.pkl'


def build_doc_embeddings():
    """
    Reads web_crawl/pages.bson and encodes each document's text into embeddings.
    Saves embeddings and doc_ids in EMBED_PATH and DOCIDS_PATH.
    """
    model = SentenceTransformer(MODEL_NAME)
    doc_ids = []
    texts = []
    # Use metadata loader to align order
    meta = load_metadata()
    id_to_text = {}
    with open(BSON_PATH, 'rb') as f:
        for doc in decode_file_iter(f):
            did = str(doc.get('_id'))
            text = doc.get('content') or doc.get('meta_description') or ''
            id_to_text[did] = text
    # Build ordered lists
    for did, title, url in meta:
        text = id_to_text.get(did, '')
        doc_ids.append(did)
        texts.append(text)
    # Encode
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    # Persist
    np.save(EMBED_PATH, embeddings)
    pickle.dump(doc_ids, open(DOCIDS_PATH, 'wb'))
    print(f"✔ Saved {len(doc_ids)} embeddings to {EMBED_PATH}")


def retrieve_semantic(query, top_k=10):
    """
    Encodes query with BERT, then retrieves top_k docs by cosine similarity over embeddings.
    Returns list of dicts: {{'doc_id','title','url','score'}}
    """
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query], convert_to_numpy=True)[0]
    # Load embeddings and doc_ids
    embeddings = np.load(EMBED_PATH)
    doc_ids = pickle.load(open(DOCIDS_PATH, 'rb'))
    # Compute cosine sims
    sims = embeddings @ q_emb / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_emb))
    idxs = np.argsort(sims)[-top_k:][::-1]
    # Load metadata for titles/urls
    meta = {did: (title, url) for did, title, url in load_metadata()}
    results = []
    for i in idxs:
        did = doc_ids[i]
        title, url = meta.get(did, ('',''))
        results.append({'doc_id': did, 'title': title, 'url': url, 'score': float(sims[i])})
    return results

if __name__ == '__main__':
    # Build embeddings if not exist
    build_doc_embeddings()  # run once
    # Quick test
    hits = retrieve_semantic('marathon training plan', top_k=5)
    for h in hits:
        print(h)
