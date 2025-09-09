from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="News NER + Dedup + Ranking API")

# Models
ner = pipeline('ner', grouped_entities=True, model='dslim/bert-base-NER')
emb_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

class Item(BaseModel):
    title: str
    text: str

class Batch(BaseModel):
    items: List[Item]
    query: str = "iphone"

@app.get("/health")
def health():
    return {"status": "ok"}

def _to_py_scalar(x):
    try:
        return x.item()
    except Exception:
        return x

def _sanitize_entities(entity_list):
    out = []
    for d in entity_list:
        dd = dict(d)
        if "score" in dd: dd["score"] = float(_to_py_scalar(dd["score"]))
        if "start" in dd: dd["start"] = int(_to_py_scalar(dd["start"]))
        if "end" in dd:   dd["end"]   = int(_to_py_scalar(dd["end"]))
        if "word" in dd:  dd["word"]  = str(dd["word"])
        if "entity_group" in dd: dd["entity_group"] = str(dd["entity_group"])
        out.append(dd)
    return out

@app.post("/process")
def process(batch: Batch):
    df = pd.DataFrame([i.dict() for i in batch.items])
    texts = df['text'].tolist()

    ents = [_sanitize_entities(ner(t)) for t in texts]
    embs = emb_model.encode(texts, normalize_embeddings=True)

    sim = cosine_similarity(embs)
    keep, removed = [], set()
    for i in range(len(df)):
        if i in removed: continue
        keep.append(i)
        for j in range(i+1, len(df)):
            if sim[i, j] >= 0.88:
                removed.add(j)

    df2 = df.iloc[keep].reset_index(drop=True)

    embs2 = emb_model.encode(df2['text'].tolist(), normalize_embeddings=True)
    q = emb_model.encode([batch.query], normalize_embeddings=True)
    qsim = cosine_similarity(embs2, q).ravel().tolist()

    ents_kept = [ents[i] for i in keep]

    payload = {
        "kept": int(len(df2)),
        "scores": [float(s) for s in qsim],
        "entities": ents_kept
    }
    return JSONResponse(content=jsonable_encoder(payload))

