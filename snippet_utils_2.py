# snippet_utils.py

import re
from bs4 import BeautifulSoup
from rake_nltk import Rake
import nltk

# Make sure you have NLTK stopwords downloaded once
nltk.download("stopwords", quiet=True)

# 1) Cleaning HTML + JS/STYLE noise
def clean_html(raw_html: str) -> str:
    """
    - Strips out <script> and <style> blocks completely
    - Removes all remaining HTML tags
    - Collapses whitespace
    - Drops any token containing digits
    """
    # Remove script/style blocks
    no_script = re.sub(r'<script[\s\S]*?</script>', ' ', raw_html, flags=re.IGNORECASE)
    no_style  = re.sub(r'<style[\s\S]*?</style>',   ' ', no_script, flags=re.IGNORECASE)

    # Use BeautifulSoup to strip tags
    text = BeautifulSoup(no_style, "html.parser").get_text(separator=" ")

    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)

    # Drop any token with a digit
    text = re.sub(r'\b\w*\d\w*\b', '', text)
    return text.strip()


# 2) RAKE extractor over the cleaned text
def extract_keywords_from_html(
    html: str,
    n_phrases: int = 5
) -> list[str]:
    """
    Returns the top `n_phrases` RAKE‐extracted keywords 
    from the cleaned HTML.
    """
    clean = clean_html(html)

    # Configure RAKE: drop very short words and limit phrase length
    rake = Rake(
        stopwords = None,   # uses NLTK stopwords by default
        min_length = 4,     # ignore words <4 chars
        max_length = 3      # ignore phrases >3 words
    )
    rake.extract_keywords_from_text(clean)
    return rake.get_ranked_phrases()[:n_phrases]


# 3) Hook into your existing snippet pipeline
def extract_snippet(html_content: str, *, n_phrases: int = 5) -> str:
    """
    Call this instead of raw RAKE on html_content.
    Joins top keywords into a single snippet string.
    """
    if not html_content:
        return ""

    keywords = extract_keywords_from_html(html_content, n_phrases=n_phrases)
    # Join with “… ” to hint these are key phrases
    return " … ".join(keywords)
