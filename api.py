# api.py
from snippet_bert import get_best_snippet

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

from retrieval_vec import retrieve_vec
from feeback_vec import retrieve_expanded
from clusters import get_top_terms, association_clusters, metric_clusters, scalar_clusters
from bson_loader_1 import load_matrix
from metadata_loader import load_metadata
from page_features import load_features

# Configuration
TOP_K      = 10
CAND_TERMS = 50
N_CLUSTERS = 4
ALPHA      = 0.7  # TF–IDF vs PageRank blend

# --- Startup: load corpus, metadata, and PageRank map ---
tfidf_matrix, doc_ids, vocab = load_matrix()
feat    = load_features()                  # dict: doc_id → pagerank
meta    = load_metadata()                  # list of (doc_id, title, url, snippet)
doc2idx = {doc_id: i for i, (doc_id, *_ ) in enumerate(meta)}

# Build feature_names array for clustering
feature_names = np.empty(len(vocab), dtype=object)
for term, idx in vocab.items():
    feature_names[idx] = term

app = FastAPI(title="IR Search + Snippet API")


# --- Pydantic models ---
class SearchHit(BaseModel):
    title:          str
    url:            str
    score:          float
    pagerank:       float = None
    combined_score: float = None
    snippet:        str


class DualRank(BaseModel):
    tfidf_only:     List[SearchHit]
    tfidf_pagerank: List[SearchHit]


# --- Helpers ---
def enrich_tf_only(hits: List[Dict], query: str) -> List[SearchHit]:
    """
    Turn raw retrieval hits into SearchHit objects,
    pulling a BERT‐selected snippet via precomputed embeddings.
    """
    out = []
    for h in hits:
        idx = doc2idx[h["doc_id"]]
        _, title, url, _static_snip, _ = meta[idx]
        # get_best_snippet(query, doc_idx) reads only the few precomputed embeddings
        dynamic_snip = get_best_snippet(query, idx)
        snippet = dynamic_snip or _static_snip
        out.append(SearchHit(
            title=title,
            url=url,
            score=h["score"],
            snippet=snippet
        ))
    return out


def rerank_with_pagerank(hits: List[Dict], query: str) -> List[SearchHit]:
    """
    Given TF–IDF hits, compute combined score with PageRank,
    and attach a BERT‐selected snippet via precomputed embeddings.
    """
    enriched = []
    for h in hits:
        pr   = feat.get(h["doc_id"], 0.0)
        comb = ALPHA * h["score"] + (1 - ALPHA) * pr
        idx = doc2idx[h["doc_id"]]
        _, title, url, _static_snip, _ = meta[idx]
        dynamic_snip = get_best_snippet(query, idx)
        snippet = dynamic_snip or _static_snip
        enriched.append(SearchHit(
            title=title,
            url=url,
            score=h["score"],
            pagerank=pr,
            combined_score=comb,
            snippet=snippet
        ))
    # sort by combined_score descending
    enriched.sort(key=lambda hit: hit.combined_score, reverse=True)
    return enriched


# --- API Endpoints ---
@app.get("/baseline", response_model=DualRank)
def baseline(query: str):
    raw_hits = retrieve_vec(query, top_k=TOP_K)
    tf_only  = enrich_tf_only(raw_hits, query)
    pr_hits  = rerank_with_pagerank(raw_hits, query)
    
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)


@app.get("/rocchio", response_model=DualRank)
def rocchio(query: str):
    exp_q, raw_hits = retrieve_expanded(query)
    tf_only = enrich_tf_only(raw_hits, exp_q)
    pr_hits = rerank_with_pagerank(raw_hits, exp_q)
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)


@app.get("/association", response_model=DualRank)
def association(query: str):
    base_hits = retrieve_vec(query, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = association_clusters(cands, rel_idxs, tfidf_matrix, feature_names)
    assoc_q   = query + " " + " ".join(terms)

    raw_hits = retrieve_vec(assoc_q, top_k=TOP_K)
    tf_only  = enrich_tf_only(raw_hits, assoc_q)
    pr_hits  = rerank_with_pagerank(raw_hits, assoc_q)
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)


@app.get("/metric", response_model=DualRank)
def metric(query: str):
    base_hits = retrieve_vec(query, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = metric_clusters(cands, rel_idxs, tfidf_matrix, feature_names, n_clusters=N_CLUSTERS)
    metric_q  = query + " " + " ".join(terms)

    raw_hits = retrieve_vec(metric_q, top_k=TOP_K)
    tf_only  = enrich_tf_only(raw_hits, metric_q)
    pr_hits  = rerank_with_pagerank(raw_hits, metric_q)
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)


@app.get("/scalar", response_model=DualRank)
def scalar(query: str):
    base_hits = retrieve_vec(query, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = scalar_clusters(cands, rel_idxs, tfidf_matrix, feature_names, n_buckets=N_CLUSTERS)
    scalar_q  = query + " " + " ".join(terms)

    raw_hits = retrieve_vec(scalar_q, top_k=TOP_K)
    tf_only  = enrich_tf_only(raw_hits, scalar_q)
    pr_hits  = rerank_with_pagerank(raw_hits, scalar_q)
    return DualRank(tfidf_only=tf_only, tfidf_pagerank=pr_hits)


@app.get("/")
def root():
    return {
        "message": "IR Search API with snippets is running.",
        "endpoints": ["/baseline", "/rocchio", "/association", "/metric", "/scalar"]
    }
