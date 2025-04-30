# bson_loader.py
import pickle
import os
from bson import decode_file_iter
from scipy import sparse

VECTORIZER_PATH = "tfidf_vectorizer.pkl"  # from Student 2
BSON_PATH       = "web_crawl/pages.bson"
MATRIX_PATH     = "doc_tfidf_matrix.npz"
DOCIDS_PATH     = "doc_ids.pkl"

def build_matrix_from_bson(
    bson_path: str = BSON_PATH,
    vectorizer_path: str = VECTORIZER_PATH,
    matrix_path: str = MATRIX_PATH,
    docids_path: str = DOCIDS_PATH
):
    # Load the existing vectorizer to get vocab → column mapping
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Missing vectorizer: {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    vocab = vectorizer.vocabulary_

    rows, cols, data, doc_ids = [], [], [], []
    with open(bson_path, "rb") as f:
        for doc_idx, doc in enumerate(decode_file_iter(f)):
            tfidf_dict = doc.get("tfidf") or {}
            if not tfidf_dict:
                continue
            doc_ids.append(str(doc["_id"]))
            for term, weight in tfidf_dict.items():
                col = vocab.get(term)
                if col is not None:
                    rows.append(len(doc_ids)-1)
                    cols.append(col)
                    data.append(weight)

    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(doc_ids), len(vocab)),
        dtype=float
    )
    sparse.save_npz(matrix_path, mat)
    with open(docids_path, "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"Built TF-IDF matrix {mat.shape}, saved to '{matrix_path}'")
    print(f"Saved {len(doc_ids)} doc IDs to '{docids_path}'")
    return mat, doc_ids

def load_corpus_matrix(
    matrix_path: str = MATRIX_PATH,
    docids_path: str = DOCIDS_PATH
):
    if not os.path.exists(matrix_path) or not os.path.exists(docids_path):
        raise FileNotFoundError("Run build_matrix_from_bson() first.")
    mat = sparse.load_npz(matrix_path)
    with open(docids_path, "rb") as f:
        doc_ids = pickle.load(f)
    if mat.shape[0] != len(doc_ids):
        raise RuntimeError("Matrix row count != number of doc IDs")
    return mat, doc_ids

if __name__ == "__main__":
    build_matrix_from_bson()
