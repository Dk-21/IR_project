import pickle
import numpy as np
from bson_loader_1 import load_matrix

IDF_PATH = "idf.npy"


def build_idf():
    mat, doc_ids, vocab = load_matrix()
    csc = mat.tocsc()
    df = np.diff(csc.indptr)
    N  = mat.shape[0]
    idf = np.log((N + 1) / (df + 1)) + 1
    np.save(IDF_PATH, idf)
    print(f"✔ Saved IDF array of length {len(idf)} to '{IDF_PATH}'")
    return idf, vocab


def load_idf():
    idf   = np.load(IDF_PATH)
    vocab = pickle.load(open("vocab.pkl", "rb"))
    return idf, vocab

if __name__=="__main__":
    build_idf()