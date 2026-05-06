"""
preprocess.py
-------------
Handles all text cleaning and preprocessing steps.
Keeps it simple: lowercase, strip punctuation, remove extra whitespace.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Clean raw input text for sentiment analysis.
    
    Steps:
      1. Lowercase everything
      2. Remove punctuation
      3. Strip extra whitespace
    
    Args:
        text: Raw customer feedback string.
    
    Returns:
        Cleaned text string.
    """
    if not text or not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation (keep spaces)
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Collapse multiple spaces into one
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_batch(texts: list[str]) -> list[str]:
    """
    Clean a list of feedback strings.
    
    Args:
        texts: List of raw feedback strings.
    
    Returns:
        List of cleaned strings.
    """
    return [clean_text(t) for t in texts]
