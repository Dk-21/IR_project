from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from urllib.parse import urlparse

from retrieval_vec import retrieve_vec
from feeback_vec import retrieve_expanded
from clusters import get_top_terms, association_clusters, metric_clusters, scalar_clusters
from bson_loader_1 import load_matrix
from metadata_loader import load_metadata
from page_features import load_features

# config
TOP_K      = 10
CAND_TERMS = 50
N_CLUSTERS = 4
ALPHA      = 0.7  # weight for TF-IDF vs PageRank in combined score

# load everything at startup
tfidf_matrix, doc_ids, vocab = load_matrix()
meta    = load_metadata()       # list of (doc_id, title, url)
feat    = load_features()       # dict doc_id->pagerank
doc2idx = {d:i for i,d in enumerate(doc_ids)}

# get feature_names array for clusters
feature_names = np.empty(len(vocab), dtype=object)
for term, idx in vocab.items():
    feature_names[idx] = term

app = FastAPI(title="IR Search + Pagerank API")

class SearchHit(BaseModel):
    title: str
    url:   str
    score: float
    pagerank: float = None
    combined_score: float = None

class DualRank(BaseModel):
    tfidf_only:      List[SearchHit]
    tfidf_pagerank:  List[SearchHit]

def rerank_with_pagerank(hits: List[Dict]) -> List[SearchHit]:
    """Given hits with 'score' and 'doc_id', compute combined_score and sort."""
    enriched = []
    for h in hits:
        pr = feat.get(h["doc_id"], 0.0)
        comb = ALPHA*h["score"] + (1-ALPHA)*pr
        enriched.append(SearchHit(
            title=h["title"],
            url=h["url"],
            score=h["score"],
            pagerank=pr,
            combined_score=comb
        ))
    # sort descending by combined_score
    enriched.sort(key=lambda x: x.combined_score, reverse=True)
    return enriched

@app.get("/baseline", response_model=DualRank)
def baseline(query: str):
    tfidf_hits = retrieve_vec(query, top_k=TOP_K)
    pr_hits    = rerank_with_pagerank(tfidf_hits)
    # convert tfidf_hits into SearchHit with pagerank=None, combined_score=None
    tf_only = [SearchHit(
        title=h["title"], url=h["url"], score=h["score"]
    ) for h in tfidf_hits]
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)

@app.get("/rocchio", response_model=DualRank)
def rocchio(query: str):
    exp_q, base_hits = retrieve_expanded(query)
    # base_hits are the expanded hits
    tfidf_hits = base_hits
    pr_hits    = rerank_with_pagerank(tfidf_hits)
    tf_only = [SearchHit(title=h["title"],url=h["url"],score=h["score"]) for h in tfidf_hits]
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)

@app.get("/association", response_model=DualRank)
def association(query: str):
    base_hits = retrieve_vec(query, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = association_clusters(cands, rel_idxs, tfidf_matrix, feature_names)
    assoc_q   = query + " " + " ".join(terms)
    tfidf_hits = retrieve_vec(assoc_q, top_k=TOP_K)
    pr_hits    = rerank_with_pagerank(tfidf_hits)
    tf_only    = [SearchHit(title=h["title"],url=h["url"],score=h["score"]) for h in tfidf_hits]
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)

@app.get("/metric", response_model=DualRank)
def metric(query: str):
    base_hits = retrieve_vec(query, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = metric_clusters(cands, rel_idxs, tfidf_matrix, feature_names, n_clusters=N_CLUSTERS)
    metric_q  = query + " " + " ".join(terms)
    tfidf_hits = retrieve_vec(metric_q, top_k=TOP_K)
    pr_hits    = rerank_with_pagerank(tfidf_hits)
    tf_only    = [SearchHit(title=h["title"],url=h["url"],score=h["score"]) for h in tfidf_hits]
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)

@app.get("/scalar", response_model=DualRank)
def scalar(query: str):
    base_hits = retrieve_vec(query, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = scalar_clusters(cands, rel_idxs, tfidf_matrix, feature_names, n_buckets=N_CLUSTERS)
    scalar_q  = query + " " + " ".join(terms)
    tfidf_hits = retrieve_vec(scalar_q, top_k=TOP_K)
    pr_hits    = rerank_with_pagerank(tfidf_hits)
    tf_only    = [SearchHit(title=h["title"],url=h["url"],score=h["score"]) for h in tfidf_hits]
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)

@app.get("/")
def root():
    return {
      "message": "Endpoints: /baseline, /rocchio, /association, /metric, /scalar",
      "note":    "Each returns `tfidf_only` and `tfidf_pagerank` rankings."
    }
