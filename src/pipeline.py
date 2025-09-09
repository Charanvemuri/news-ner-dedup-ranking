import argparse
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import pipeline

def run_ner(texts):
    nlp = pipeline('ner', grouped_entities=True, model='dslim/bert-base-NER')
    return [nlp(t) for t in texts]

def embed(texts):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(texts, convert_to_tensor=False, normalize_embeddings=True)

def deduplicate(df, embeddings, threshold=0.88):
    sim = cosine_similarity(embeddings)
    keep = []
    removed = set()
    for i in range(len(df)):
        if i in removed:
            continue
        keep.append(i)
        for j in range(i+1, len(df)):
            if sim[i, j] >= threshold:
                removed.add(j)
    return df.iloc[keep].reset_index(drop=True), sim

def score(df, embeddings, query="iphone"):
    # toy scoring: similarity to query + keyword presence
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    q_emb = model.encode([query], normalize_embeddings=True)
    sim = cosine_similarity(embeddings, q_emb).ravel()
    keyword = df['text'].str.contains(query, case=False).astype(int)
    return 0.7 * sim + 0.3 * keyword

def main(args):
    df = pd.read_csv(args.input)
    texts = df['text'].fillna("").tolist()
    ents = run_ner(texts)
    embs = embed(texts)
    df_dedup, _ = deduplicate(df, embs, threshold=0.88)
    embs_dedup = embed(df_dedup['text'].tolist())
    df_dedup['score'] = score(df_dedup, embs_dedup, query="iphone")
    df_dedup['entities'] = run_ner(df_dedup['text'].tolist())
    df_dedup.sort_values('score', ascending=False, inplace=True)
    df_dedup.to_csv(args.output, index=False)
    print(f"Saved results to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sample_news.csv")
    parser.add_argument("--output", default="out/results.csv")
    args = parser.parse_args()
    import os
    os.makedirs("out", exist_ok=True)
    main(args)