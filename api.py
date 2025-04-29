# api.py
from snippet_bert import get_best_snippet
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import numpy as np
from query_vector import clean_query
from feedback_rake import retrieve_expanded_rake
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



class SearchHit(BaseModel):
    title:          str
    url:            str
    score:          float
    pagerank:       Optional[float] = None
    combined_score: Optional[float] = None
    snippet:        str

class DualRank(BaseModel):
    expanded_query: str
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
# api.py (excerpt)

@app.get("/baseline", response_model=DualRank)
def baseline(query: str):
    # baseline has no expansion, so exp_q = original
    clean_q= clean_query(query)
    exp_q     = clean_q
    raw_hits  = retrieve_vec(exp_q, top_k=TOP_K)
    tf_only   = enrich_tf_only(raw_hits, exp_q)
    pr_hits   = rerank_with_pagerank(raw_hits, exp_q)
    return DualRank(
        expanded_query=exp_q,
        tfidf_only=tf_only,
        tfidf_pagerank=pr_hits,
    )


@app.get("/rocchio", response_model=DualRank)
def rocchio(query: str):
    # retrieve_expanded returns (expanded_query, hits)
    clean_q= clean_query(query)
    exp_q, raw_hits = retrieve_expanded_rake(clean_q)
    clean_q=clean_query(exp_q)
    tf_only         = enrich_tf_only(raw_hits, clean_q)
    pr_hits         = rerank_with_pagerank(raw_hits, clean_q)
    return DualRank(
        expanded_query=clean_q,
        tfidf_only=tf_only,
        tfidf_pagerank=pr_hits,
    )


@app.get("/association", response_model=DualRank)
def association(query: str):
    clean_q= clean_query(query)
    # 1) get top terms & build association‐expanded query
    base_hits = retrieve_vec(clean_q , top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = association_clusters(cands, rel_idxs, tfidf_matrix, feature_names)
    exp_q     = query + " " + " ".join(terms)

    # 2) retrieve on exp_q
    clean_q = clean_query(exp_q)
    raw_hits = retrieve_vec(clean_q, top_k=TOP_K)
    tf_only  = enrich_tf_only(raw_hits, clean_q)
    pr_hits  = rerank_with_pagerank(raw_hits, clean_q)
    return DualRank(
        expanded_query=clean_q,
        tfidf_only=tf_only,
        tfidf_pagerank=pr_hits,
    )


@app.get("/metric", response_model=DualRank)
def metric(query: str):
    clean_q = clean_query(query)
    base_hits = retrieve_vec(clean_q, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = metric_clusters(cands, rel_idxs, tfidf_matrix, feature_names, n_clusters=N_CLUSTERS)
    exp_q     = clean_q + " " + " ".join(terms)
    clean_q = clean_query(exp_q)
    raw_hits = retrieve_vec(clean_q, top_k=TOP_K)
    tf_only  = enrich_tf_only(raw_hits, clean_q)
    pr_hits  = rerank_with_pagerank(raw_hits, clean_q)
    return DualRank(
        expanded_query=clean_q,
        tfidf_only=tf_only,
        tfidf_pagerank=pr_hits,
    )


@app.get("/scalar", response_model=DualRank)
def scalar(query: str):
    clean_q = clean_query(query)
    base_hits = retrieve_vec(clean_q, top_k=TOP_K)
    rel_idxs  = [doc2idx[h["doc_id"]] for h in base_hits]
    cands     = get_top_terms(rel_idxs, tfidf_matrix, feature_names, top_n=CAND_TERMS)
    terms     = scalar_clusters(cands, rel_idxs, tfidf_matrix, feature_names, n_buckets=N_CLUSTERS)
    exp_q     = clean_q + " " + " ".join(terms)
    clean_q = clean_query(exp_q)
    raw_hits = retrieve_vec(clean_q, top_k=TOP_K)
    tf_only  = enrich_tf_only(raw_hits, clean_q)
    pr_hits  = rerank_with_pagerank(raw_hits, clean_q)
    return DualRank(
        expanded_query=clean_q,
        tfidf_only=tf_only,
        tfidf_pagerank=pr_hits,
    )



@app.get("/")
def root():
    return {
        "message": "IR Search API with snippets is running.",
        "endpoints": ["/baseline", "/rocchio", "/association", "/metric", "/scalar"]
    }
