"""
Text preprocessing module for DrivebuddyAI Pav Bhaji Challenge.
Implements robust cleaning, normalization, emoji handling,
and leakage-controlled text representations.
"""

import re
import string
import unicodedata
from typing import Optional, List, Set

# Regex patterns for cleaning
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
MENTION_PATTERN = re.compile(r'@\w+')
HTML_TAG_PATTERN = re.compile(r'<.*?>')
EMOJI_PATTERN = re.compile(
    r'[\U00010000-\U0010ffff]'
    r'|[\u2600-\u27bf]'
    r'|[\u2300-\u23ff]'
    r'|[\u2b50-\u2b55]'
    r'|[\u200d]'
    r'|[\ufe0f]',
    flags=re.UNICODE
)

# Target leakage patterns: explicitly identifying pav bhaji variants
TARGET_LEAKAGE_PATTERNS = [
    re.compile(r'#?pav[\s\-_]*bhaj[ij]+[a-z]*', re.IGNORECASE),
    re.compile(r'#?bhaj[ij]+[\s\-_]*pav[a-z]*', re.IGNORECASE),
    re.compile(r'\bpav[\s\-_]*bhaj[ij]+\b', re.IGNORECASE),
]

# Common English and social media stopwords (preserving food terms)
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def remove_target_leakage_terms(text: str) -> str:
    """
    Remove or mask explicit Pav Bhaji target terms to evaluate genuine semantic
    generalization (Experiment 2: Leakage-Controlled).
    """
    cleaned = text
    for pattern in TARGET_LEAKAGE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    # Also strip standalone tokens if they directly reveal target
    cleaned = re.sub(r'\bpav\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bbhaji\b', ' ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\bbhajji\b', ' ', cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_hashtags(text: str) -> str:
    """Normalize hashtag symbols into words while separating concatenated words."""
    # Replace #word with word
    return re.sub(r'#(\w+)', r'\1', text)


def preprocess_text(
    text: str,
    remove_leakage: bool = False,
    remove_stopwords: bool = False,
    stopwords: Optional[Set[str]] = None,
    keep_hashtags: bool = True
) -> str:
    """
    Comprehensive text preprocessing pipeline for Instagram captions & tags.
    
    Args:
        text: Raw input string.
        remove_leakage: If True, strips direct target keywords ('pav bhaji', '#pavbhaji').
        remove_stopwords: If True, removes standard English function words.
        stopwords: Custom set of stopwords, defaults to DEFAULT_STOPWORDS.
        keep_hashtags: If True, converts #tag to tag; if False, strips tags completely.
    
    Returns:
        Cleaned, normalized string.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Normalize unicode characters (NFKD)
    cleaned = unicodedata.normalize("NFKD", text)

    # 2. Lowercase
    cleaned = cleaned.lower()

    # 3. Strip URLs
    cleaned = URL_PATTERN.sub(" ", cleaned)

    # 4. Strip HTML tags
    cleaned = HTML_TAG_PATTERN.sub(" ", cleaned)

    # 5. Strip Instagram mentions (@user)
    cleaned = MENTION_PATTERN.sub(" ", cleaned)

    # 6. Leakage control (if requested for Experiment 2)
    if remove_leakage:
        cleaned = remove_target_leakage_terms(cleaned)

    # 7. Normalize or strip hashtags
    if keep_hashtags:
        cleaned = clean_hashtags(cleaned)
    else:
        cleaned = re.sub(r'#\w+', ' ', cleaned)

    # 8. Remove emojis / symbols
    cleaned = EMOJI_PATTERN.sub(" ", cleaned)

    # 9. Clean punctuation & special characters (preserve letters and digits)
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', ' ', cleaned)

    # 10. Normalize whitespace
    tokens = cleaned.split()

    # 11. Optional stopword removal
    if remove_stopwords:
        sw = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        tokens = [t for t in tokens if t not in sw and len(t) > 1]
    else:
        tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens)


if __name__ == "__main__":
    sample = "TAG A PAV BHAJI FANATIC 😋😍 PC @exploringplates #pavbhaji #mumbaifoodie http://insta.com"
    print("Original:         ", sample)
    print("Preprocessed (Raw):", preprocess_text(sample, remove_leakage=False))
    print("Preprocessed (No Leak):", preprocess_text(sample, remove_leakage=True))
