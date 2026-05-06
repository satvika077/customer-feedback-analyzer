# Customer Feedback Analyzer

A small ML project I built to practice working with NLP pipelines and LLM APIs. It takes in customer reviews or comments and outputs two things — the sentiment (positive, negative, or neutral) and a short summary of the key points.

---

## What it does

- Cleans and preprocesses raw text input
- Runs sentiment analysis using TextBlob (polarity scoring)
- Calls the Claude API to generate a 1–2 sentence summary
- Exposes everything through a simple FastAPI endpoint

---

## Project Structure

```
project/
├── data/
│   ├── feedback.csv       # sample reviews I used for testing
│   └── results.json       # output from running the batch pipeline
├── src/
│   ├── main.py            # runs the full pipeline on the CSV
│   ├── preprocess.py      # text cleaning (lowercase, strip punctuation, etc.)
│   ├── model.py           # sentiment analysis logic
│   ├── llm_utils.py       # Claude API call for summarization
│   └── api.py             # FastAPI app with the /analyze endpoint
├── requirements.txt
└── README.md
```

---

## Setup

**1. Clone the repo and go into the folder**

```bash
git clone https://github.com/YOUR_USERNAME/customer-feedback-analyzer.git
cd customer-feedback-analyzer
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
python -m textblob.download_corpora
```

**4. Add your Anthropic API key**

Create a `.env` file in the root folder:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Or just export it directly in your terminal:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running it

**Option A — batch mode (processes the whole CSV)**

```bash
python src/main.py
```

Reads from `data/feedback.csv` and saves results to `data/results.json`.

**Option B — API mode**

```bash
uvicorn src.api:app --reload
```

Runs at `http://127.0.0.1:8000`. You can also open `http://127.0.0.1:8000/docs` for the interactive Swagger UI.

---

## API

**POST** `/analyze`

```json
{
  "text": "The product quality is amazing! Fast shipping and great packaging."
}
```

**Response:**

```json
{
  "sentiment": "positive",
  "score": 0.625,
  "confidence": "high",
  "summary": "The customer is highly satisfied with the product quality, shipping speed, and packaging.",
  "original_text": "The product quality is amazing! Fast shipping and great packaging."
}
```

**cURL example:**

```bash
curl -X POST http://127.0.0.1:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Terrible product, broke after one use. Very disappointed."}'
```

**A few more examples:**

| Input | Sentiment | Score |
|-------|-----------|-------|
| "Amazing product, super fast shipping!" | positive | 0.625 |
| "Broke after one day. Waste of money." | negative | -0.45 |
| "It works fine, nothing special." | neutral | 0.05 |

---

## How the pipeline works

```
Raw text input
      │
      ▼
preprocess.py  →  lowercase, remove punctuation, strip whitespace
      │
      ▼
model.py       →  TextBlob polarity score → sentiment + confidence level
      │
      ▼
llm_utils.py   →  Claude API → 1-2 sentence summary
      │
      ▼
{ sentiment, score, confidence, summary }
```

I used TextBlob for sentiment because it's lightweight, needs no training data, and the polarity score is easy to interpret and debug. I used Claude (Haiku model) for summarization because it handles messy, informal review language really well without needing much prompt engineering.

---

## Things I'd improve with more time

- Swap TextBlob for a fine-tuned model like `distilbert-base-uncased-finetuned-sst-2-english` for better accuracy
- Add a `/batch` endpoint that accepts multiple reviews at once
- Store results in a SQLite database
- Build a simple frontend so non-technical users can use it
