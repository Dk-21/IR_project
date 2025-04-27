import pickle, os
from bson import decode_file_iter
from scipy import sparse

BSON_PATH    = "web_crawl/pages.bson"
MATRIX_PATH  = "doc_tfidf_matrix.npz"
DOCIDS_PATH  = "doc_ids.pkl"
VOCAB_PATH   = "vocab.pkl"


def build_matrix():
    rows, cols, data, doc_ids = [], [], [], []
    vocab = {}
    next_col = 0

    with open(BSON_PATH, "rb") as f:
        for doc in decode_file_iter(f):
            tf = doc.get("tfidf") or {}
            if not tf:
                continue
            doc_ids.append(str(doc["_id"]))
            for term, w in tf.items():
                if term not in vocab:
                    vocab[term] = next_col
                    next_col += 1
                col = vocab[term]
                rows.append(len(doc_ids)-1)
                cols.append(col)
                data.append(w)

    mat = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(len(doc_ids), next_col),
        dtype=float
    )
    sparse.save_npz(MATRIX_PATH, mat)
    pickle.dump(doc_ids, open(DOCIDS_PATH, "wb"))
    pickle.dump(vocab, open(VOCAB_PATH, "wb"))
    print(f"✔ Built TF-IDF matrix {mat.shape} → '{MATRIX_PATH}'")
    return mat, doc_ids, vocab


def load_matrix():
    mat     = sparse.load_npz(MATRIX_PATH)
    doc_ids = pickle.load(open(DOCIDS_PATH, "rb"))
    vocab   = pickle.load(open(VOCAB_PATH, "rb"))
    return mat, doc_ids, vocab

if __name__=="__main__":
    build_matrix()