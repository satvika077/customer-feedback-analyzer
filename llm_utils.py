"""
llm_utils.py
------------
Uses the Anthropic Claude API to generate a concise summary
of customer feedback text.

Keep it minimal: one prompt, one function, clear output.
"""

import os
import anthropic


# Initialize the Anthropic client once at module load.
# It reads ANTHROPIC_API_KEY from the environment automatically.
_client = anthropic.Anthropic()

# The model to use for summarization
MODEL = "claude-haiku-4-5-20251001"  # Fast and cost-effective for short tasks


def summarize_feedback(text: str, sentiment: str) -> str:
    """
    Use Claude to generate a 1-2 sentence summary of customer feedback.

    The prompt provides the sentiment context so Claude can tailor
    the summary tone (e.g., highlight pain points for negative reviews).

    Args:
        text      : The original (uncleaned) customer feedback.
        sentiment : "positive", "negative", or "neutral"

    Returns:
        A short summary string (1-2 sentences).
    """
    prompt = f"""You are a customer feedback analyst.

Summarize the following customer feedback in 1-2 sentences.
Focus on the key points and the main reason for the customer's sentiment.
Be concise and neutral in tone.

Sentiment detected: {sentiment}
Feedback: "{text}"

Summary:"""

    message = _client.messages.create(
        model=MODEL,
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the text from the response
    summary = message.content[0].text.strip()
    return summary


def summarize_batch(texts: list[str], sentiments: list[str]) -> list[str]:
    """
    Summarize a list of feedback texts with their corresponding sentiments.

    Args:
        texts      : List of raw feedback strings.
        sentiments : List of sentiment labels (same length as texts).

    Returns:
        List of summary strings.
    """
    summaries = []
    for text, sentiment in zip(texts, sentiments):
        summary = summarize_feedback(text, sentiment)
        summaries.append(summary)
    return summaries
