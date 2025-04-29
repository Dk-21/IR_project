# snippet_utils.py

import re

# Regex to split on sentence-ending punctuation (., !, ?) plus whitespace
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+')

def sent_tokenize_regex(text: str):
    """
    Very lightweight sentence tokenizer: splits on ". ", "! ", "? ".
    """
    if not text:
        return []
    # Split, keeping the punctuation at end of each sentence
    parts = _SENTENCE_END_RE.split(text.strip())
    # Filter out empty results
    return [p.strip() for p in parts if p.strip()]

def extract_snippet(text: str, query: str = None, min_length: int = 150) -> str:
    """
    Heuristic snippet extractor WITHOUT NLTK:
      1. If query appears, return the first sentence containing it.
      2. Else pick the middle sentence (or join two if too short).
      3. Fallback to the first min_length chars of text.
    """
    t = text.strip()
    if not t:
        return ""

    sents = sent_tokenize_regex(t)

    # 1) Query match
    if query:
        q = query.lower()
        for s in sents:
            if q in s.lower():
                return s if len(s) >= min_length else s + "…"

    # 2) Middle sentence
    if sents:
        mid = len(sents) // 2
        snippet = sents[mid]
        # if too short, append next
        if len(snippet) < min_length and mid+1 < len(sents):
            snippet = snippet + " " + sents[mid+1]
        return snippet if len(snippet) >= min_length else snippet + "…"

    # 3) Fallback first min_length chars
    return (t[:min_length] + "…") if len(t) > min_length else t
