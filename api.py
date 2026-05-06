"""
api.py
------
FastAPI app exposing a single /analyze endpoint.

Start the server:
    uvicorn src.api:app --reload

Then POST to:
    http://127.0.0.1:8000/analyze
"""

import os
import sys

# Allow src-relative imports
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from preprocess import clean_text
from model import analyze_sentiment
from llm_utils import summarize_feedback


app = FastAPI(
    title="Customer Feedback Analyzer",
    description="Analyzes customer feedback: returns sentiment + LLM-generated summary.",
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    text: str  # Raw customer feedback


class FeedbackResponse(BaseModel):
    sentiment: str      # "positive" | "negative" | "neutral"
    score: float        # Polarity score from -1.0 to +1.0
    confidence: str     # "high" | "medium" | "low"
    summary: str        # LLM-generated 1-2 sentence summary
    original_text: str  # Echo back the input


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "Customer Feedback Analyzer is running."}


@app.post("/analyze", response_model=FeedbackResponse)
def analyze(request: FeedbackRequest):
    """
    Analyze a single piece of customer feedback.

    - **text**: The raw customer review or comment.

    Returns sentiment label, polarity score, confidence level,
    and a short LLM-generated summary.
    """
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if len(text) > 2000:
        raise HTTPException(status_code=400, detail="Text too long. Max 2000 characters.")

    # Step 1: Preprocess
    cleaned = clean_text(text)

    # Step 2: Sentiment analysis
    sentiment_result = analyze_sentiment(cleaned)

    # Step 3: LLM summary (use original text for richer context)
    summary = summarize_feedback(text, sentiment_result["sentiment"])

    return FeedbackResponse(
        sentiment=sentiment_result["sentiment"],
        score=sentiment_result["score"],
        confidence=sentiment_result["confidence"],
        summary=summary,
        original_text=text,
    )
