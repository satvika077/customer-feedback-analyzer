"""
model.py
--------
Sentiment analysis using a lightweight pre-trained model.
Uses TextBlob for polarity scoring — no GPU or large downloads required.
Returns: "positive", "negative", or "neutral"
"""

from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of a single text string using TextBlob polarity.

    TextBlob returns a polarity score in [-1.0, +1.0]:
      - Positive  → polarity >  0.1
      - Negative  → polarity < -0.1
      - Neutral   → everything in between

    Args:
        text: Cleaned or raw customer feedback string.

    Returns:
        dict with keys:
            - "sentiment" : "positive" | "negative" | "neutral"
            - "score"     : float polarity score
            - "confidence": "high" | "medium" | "low" (based on magnitude)
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity        # -1.0 to +1.0
    subjectivity = blob.sentiment.subjectivity  # 0.0 to 1.0

    # Classify based on thresholds
    if polarity > 0.1:
        label = "positive"
    elif polarity < -0.1:
        label = "negative"
    else:
        label = "neutral"

    # Simple confidence based on how far from 0 the score is
    abs_score = abs(polarity)
    if abs_score >= 0.5:
        confidence = "high"
    elif abs_score >= 0.2:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "sentiment": label,
        "score": round(polarity, 4),
        "subjectivity": round(subjectivity, 4),
        "confidence": confidence,
    }


def analyze_batch(texts: list[str]) -> list[dict]:
    """
    Run sentiment analysis on a list of texts.

    Args:
        texts: List of feedback strings.

    Returns:
        List of sentiment result dicts.
    """
    return [analyze_sentiment(t) for t in texts]
