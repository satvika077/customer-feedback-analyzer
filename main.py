"""
main.py
-------
End-to-end pipeline: load data → preprocess → sentiment → summarize → print results.

Run this directly to process the sample CSV dataset:
    python src/main.py
"""

import csv
import json
import os
import sys

# Allow imports from src/ when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import clean_text
from model import analyze_sentiment
from llm_utils import summarize_feedback


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "feedback.csv")


def load_csv(filepath: str) -> list[dict]:
    """
    Load feedback rows from a CSV file.
    Expected columns: id, text

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of dicts with 'id' and 'text' keys.
    """
    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({"id": row["id"], "text": row["text"]})
    return rows


def analyze_feedback(text: str) -> dict:
    """
    Full pipeline for a single feedback string.

    Steps:
        1. Clean/preprocess the text
        2. Run sentiment analysis (TextBlob)
        3. Generate summary via LLM

    Args:
        text: Raw feedback string.

    Returns:
        dict with sentiment info and summary.
    """
    # Step 1: Preprocess (clean for sentiment model)
    cleaned = clean_text(text)

    # Step 2: Sentiment analysis
    sentiment_result = analyze_sentiment(cleaned)

    # Step 3: LLM summarization (use original text for better context)
    summary = summarize_feedback(text, sentiment_result["sentiment"])

    return {
        "original_text": text,
        "sentiment": sentiment_result["sentiment"],
        "score": sentiment_result["score"],
        "confidence": sentiment_result["confidence"],
        "summary": summary,
    }


def run_pipeline():
    """
    Load dataset and run the full analysis pipeline on each row.
    Prints results to stdout as formatted JSON.
    """
    print("=" * 60)
    print("  Customer Feedback Analysis Pipeline")
    print("=" * 60)

    # Load data
    rows = load_csv(DATA_PATH)
    print(f"\nLoaded {len(rows)} feedback entries from CSV.\n")

    results = []
    for row in rows:
        print(f"[ID {row['id']}] Analyzing...")
        result = analyze_feedback(row["text"])
        result["id"] = row["id"]
        results.append(result)

        # Print a clean summary per row
        print(f"  Text      : {row['text'][:70]}...")
        print(f"  Sentiment : {result['sentiment'].upper()} "
              f"(score: {result['score']}, confidence: {result['confidence']})")
        print(f"  Summary   : {result['summary']}")
        print()

    # Save results to JSON
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
