
from fastapi import FastAPI, HTTPException
import pickle, numpy as np, pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Paper Recommendation API", version="1.0.0")

# Load artifacts on startup
df = pd.read_parquet("papers.parquet")
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
tfidf_matrix = np.load("tfidf_matrix.npy")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/health")
def health():
    return {"status": "ok", "papers": len(df)}

@app.get("/recommend/{paper_idx}")
def recommend(paper_idx: int, k: int = 10):
    if paper_idx < 0 or paper_idx >= len(df):
        raise HTTPException(status_code=404, detail="Paper not found")
    query_vec = tfidf_matrix[paper_idx]
    scores = cosine_similarity([query_vec], tfidf_matrix).flatten()
    scores[paper_idx] = -1
    top_k = np.argsort(scores)[::-1][:k]
    return {
        "query": df.loc[paper_idx, "title"],
        "recommendations": [
            {"rank": i+1, "title": df.loc[idx, "title"],
             "score": round(float(scores[idx]), 4)}
            for i, idx in enumerate(top_k)
        ]
    }

@app.get("/stats")
def stats():
    return {"total_papers": len(df), "categories": df["primary_category"].nunique()}
