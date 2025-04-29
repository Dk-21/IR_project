# feedback_rake.py

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)

from rake_nltk import Rake
from retrieval_vec import retrieve_vec
from query_vector import clean_query

# RAKE parameters
TOP_K            = 10     # must match your other endpoints
RAKE_MAX_PHRASES = 15   # how many candidate phrases to extract
EXP_TERMS        = 10   # how many to keep in your final expansion

# Initialize RAKE once: English stopwords + punctuation
rake = Rake()

def rake_expand(query: str) -> str:
    """
    1) Clean the user query
    2) Retrieve TOP_K docs via retrieve_vec()
    3) Concatenate their text content (you could read from your meta store)
    4) Run RAKE to extract top phrases
    5) Pick the top EXP_TERMS single words from those phrases
    6) Return "query + expansions"
    """
    # 1) Clean
    cq = clean_query(query)

    # 2) Get the top‐K relevant hits
    hits = retrieve_vec(cq, top_k=TOP_K)

    # 3) Collect their snippets or full text
    #    Here we assume each hit dict has a 'snippet' field with ~100 chars
    #    If you want longer text, pull it from your metadata instead.
    docs_text = " ".join(h["snippet"] for h in hits if h.get("snippet"))

    if not docs_text:
        return query

    # 4) Run RAKE on the concatenated text
    rake.extract_keywords_from_text(docs_text)
    phrase_scores = rake.get_ranked_phrases_with_scores()  # list of (score, phrase)

    # 5) Flatten phrases into words, preserving RAKE score order
    candidates = []
    for score, phrase in phrase_scores[:RAKE_MAX_PHRASES]:
        for w in phrase.split():
            w = w.lower()
            if w.isalpha() and len(w) >= 4 and w not in cq:
                candidates.append((score, w))

    # Deduplicate while preserving order
    seen = set()
    terms = []
    for score, w in sorted(candidates, key=lambda x: -x[0]):
        if w not in seen:
            seen.add(w)
            terms.append(w)
        if len(terms) >= EXP_TERMS:
            break

    # 6) Return the final expansion
    if terms:
        return query + " " + " ".join(terms)
    else:
        return query


def retrieve_expanded_rake(query: str):
    """
    The same API shape as before: returns (expanded_q, hits).
    """
    exp_q = rake_expand(query)
    hits  = retrieve_vec(exp_q, top_k=TOP_K)
    return exp_q, hits
