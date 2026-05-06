# Customer Feedback Analyzer

A clean, end-to-end ML project that analyzes customer feedback and returns:
- **Sentiment** — positive, negative, or neutral (via TextBlob)
- **Summary** — a concise 1–2 sentence summary (via Claude LLM)

Built as a practical ML internship demo: clean code, simple pipeline, real LLM integration.

---

## Project Structure

```
project/
├── data/
│   ├── feedback.csv       # Sample customer feedback dataset
│   └── results.json       # Output after running the pipeline
├── src/
│   ├── main.py            # End-to-end pipeline (batch mode)
│   ├── preprocess.py      # Text cleaning utilities
│   ├── model.py           # Sentiment analysis (TextBlob)
│   ├── llm_utils.py       # LLM summarization (Anthropic Claude)
│   └── api.py             # FastAPI app with /analyze endpoint
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / download the project

```bash
cd project
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # Download TextBlob data
```

### 4. Set your Anthropic API key

Create a `.env` file or export directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## How to Run

### Option A — Batch Pipeline (CSV → JSON results)

```bash
python src/main.py
```

Processes all rows in `data/feedback.csv` and saves output to `data/results.json`.

### Option B — FastAPI Server

```bash
uvicorn src.api:app --reload
```

Server runs at `http://127.0.0.1:8000`

Interactive API docs: `http://127.0.0.1:8000/docs`

---

## API Usage

### Endpoint

```
POST /analyze
Content-Type: application/json
```

### Request body

```json
{
  "text": "The product quality is amazing! Fast shipping and great packaging."
}
```

### Response

```json
{
  "sentiment": "positive",
  "score": 0.625,
  "confidence": "high",
  "summary": "The customer is highly satisfied with the product quality, shipping speed, and packaging.",
  "original_text": "The product quality is amazing! Fast shipping and great packaging."
}
```

### cURL example

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible product, broke after one use. Very disappointed."}'
```

### Example outputs

| Input | Sentiment | Score | Summary |
|-------|-----------|-------|---------|
| "Amazing product, super fast shipping!" | positive | 0.625 | The customer is highly satisfied with both the product and delivery speed. |
| "Broke after one day. Waste of money." | negative | -0.45 | The customer reports product failure and considers the purchase a poor value. |
| "It works fine, nothing special." | neutral | 0.05 | The customer finds the product acceptable but unremarkable. |

---

## How It Works

```
Raw Text
   │
   ▼
preprocess.py  →  Lowercase, remove punctuation, strip whitespace
   │
   ▼
model.py       →  TextBlob polarity score → sentiment label + confidence
   │
   ▼
llm_utils.py   →  Claude API prompt → 1-2 sentence summary
   │
   ▼
Output: { sentiment, score, confidence, summary }
```

### Why TextBlob for sentiment?
- Zero setup, no GPU, no training data needed
- Good enough accuracy for common feedback patterns
- Transparent and debuggable (polarity score is human-readable)

### Why Claude for summarization?
- Handles nuanced language far better than extractive approaches
- One simple prompt, minimal token usage (Haiku model)
- Easy to swap for any other LLM provider

---

## Extending This Project

- **Swap TextBlob** → use `transformers` with `distilbert-base-uncased-finetuned-sst-2-english` for higher accuracy
- **Add a database** → store results in SQLite with `sqlite3`
- **Batch endpoint** → accept a list of texts in one API call
- **Frontend** → add a simple HTML form using `Jinja2` templates
