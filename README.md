# News NER + Dedup + Simple Ranking

This project extracts entities from news articles, embeds articles for semantic similarity,
removes near-duplicates, and ranks remaining items by a basic relevance score.

## Components
- **NER:** Hugging Face pipeline (`dslim/bert-base-NER`) by default
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
- **Deduplication:** Cosine similarity thresholding
- **Ranking:** Simple BM25-like score + embedding similarity (toy example)

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py --input data/sample_news.csv --output out/results.csv
```

## API (optional)
```bash
uvicorn src.api:app --reload --port 8002
```

## Files
- `data/sample_news.csv` - tiny sample dataset
- `src/pipeline.py` - CLI to run NER, dedup, ranking
- `src/api.py` - FastAPI to run pipeline in-memory
- `requirements.txt` - dependencies

## Notes
- On first run, Transformers/SentenceTransformers will download models.
- Adjust deduplication threshold in `pipeline.py` as needed (default 0.88).