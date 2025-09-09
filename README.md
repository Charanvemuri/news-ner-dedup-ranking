# 📰 News NER + Dedup + Ranking API

A lightweight **NLP pipeline and API** for processing news articles.  
It extracts entities, removes near-duplicate articles, and ranks remaining items by relevance to a query.

---

## 🚀 Components
- **NER:** Hugging Face `dslim/bert-base-NER`
- **Embeddings:** SentenceTransformers `all-MiniLM-L6-v2`
- **Deduplication:** Cosine similarity thresholding
- **Ranking:** Query embedding similarity

---

## ▶️ Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py --input data/sample_news.csv --output out/results.csv

uvicorn src.api:app --reload --port 8002

curl -sS -X POST "http://127.0.0.1:8002/process" \
  -H "Content-Type: application/json" \
  -d '{"query":"iphone","items":[{"title":"A","text":"Apple announced new iPhone with better cameras."}]}'


---

### 🔄 Step 2 — Commit and push
```bash
git add README.md
git commit -m "docs: polished README with API usage and highlights"
git push

git add README.md
git commit -m "docs: polished README with API usage and highlights"
git push

