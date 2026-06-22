
from fastapi import FastAPI, HTTPException
import pickle, numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os

app = FastAPI(title="Paper Recommendation API", version="1.0.0")

# Download large files from HF dataset hub if not present
DATA_REPO = "shahwar98/paper-recommender-data"

if not os.path.exists("papers.parquet"):
    print("Downloading papers.parquet...")
    hf_hub_download(repo_id=DATA_REPO, filename="papers.parquet",
                    repo_type="dataset", local_dir=".")

if not os.path.exists("tfidf_vectorizer.pkl"):
    print("Downloading tfidf_vectorizer.pkl...")
    hf_hub_download(repo_id=DATA_REPO, filename="tfidf_vectorizer.pkl",
                    repo_type="dataset", local_dir=".")

if not os.path.exists("tfidf_matrix_sparse.npz"):
    print("Downloading tfidf_matrix_sparse.npz...")
    hf_hub_download(repo_id=DATA_REPO, filename="tfidf_matrix_sparse.npz",
                    repo_type="dataset", local_dir=".")

# Load artifacts
print("Loading artifacts...")
df = pd.read_parquet("papers.parquet")
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
tfidf_matrix = sp.load_npz("tfidf_matrix_sparse.npz")
st_model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Ready — {len(df):,} papers loaded")

@app.get("/health")
def health():
    return {"status": "ok", "papers": len(df)}

@app.get("/recommend/{paper_idx}")
def recommend(paper_idx: int, k: int = 10, method: str = "hybrid"):
    if paper_idx < 0 or paper_idx >= len(df):
        raise HTTPException(status_code=404, detail="Paper not found")

    query_vec = tfidf_matrix[paper_idx]
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    scores[paper_idx] = -1
    top_50 = np.argsort(scores)[::-1][:50]

    if method == "tfidf":
        top_k = top_50[:k]
        return {
            "query": {"idx": paper_idx, "title": df.loc[paper_idx, "title"],
                      "category": df.loc[paper_idx, "primary_category"]},
            "method": "tfidf",
            "recommendations": [
                {"rank": i+1, "title": df.loc[int(idx), "title"],
                 "category": df.loc[int(idx), "primary_category"],
                 "score": round(float(scores[idx]), 4)}
                for i, idx in enumerate(top_k)
            ]
        }

    query_text = df.loc[paper_idx, "combined_text"]
    cand_texts = [df.loc[int(i), "combined_text"] for i in top_50]
    embeddings = st_model.encode([query_text] + cand_texts,
                                  normalize_embeddings=True, batch_size=32)
    query_emb = embeddings[0]
    cand_embs = embeddings[1:]
    sem_scores = (cand_embs @ query_emb).tolist()

    results = []
    for i, idx in enumerate(top_50):
        results.append({
            "rank": 0,
            "title": df.loc[int(idx), "title"],
            "category": df.loc[int(idx), "primary_category"],
            "score": round(0.3 * float(scores[idx]) + 0.7 * sem_scores[i], 4)
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results[:k]):
        r["rank"] = i + 1

    return {
        "query": {"idx": paper_idx, "title": df.loc[paper_idx, "title"],
                  "category": df.loc[paper_idx, "primary_category"]},
        "method": "hybrid",
        "recommendations": results[:k]
    }

@app.get("/paper/{paper_idx}")
def get_paper(paper_idx: int):
    if paper_idx < 0 or paper_idx >= len(df):
        raise HTTPException(status_code=404, detail="Paper not found")
    row = df.loc[paper_idx]
    return {"idx": paper_idx, "arxiv_id": row["id"], "title": row["title"],
            "abstract": row["abstract"][:500] + "...",
            "category": row["primary_category"],
            "year": int(row["year"]) if pd.notna(row["year"]) else None}

@app.get("/stats")
def stats():
    return {"total_papers": len(df),
            "categories": df["primary_category"].nunique(),
            "top_categories": df["primary_category"].value_counts().head(5).to_dict()}
