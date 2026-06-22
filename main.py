
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import pickle, numpy as np, pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import os

app = FastAPI(
    title="Paper Recommendation API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

DATA_REPO = "shahwar98/paper-recommender-data"

if not os.path.exists("papers.parquet"):
    hf_hub_download(repo_id=DATA_REPO, filename="papers.parquet",
                    repo_type="dataset", local_dir=".")
if not os.path.exists("tfidf_vectorizer.pkl"):
    hf_hub_download(repo_id=DATA_REPO, filename="tfidf_vectorizer.pkl",
                    repo_type="dataset", local_dir=".")
if not os.path.exists("tfidf_matrix_sparse.npz"):
    hf_hub_download(repo_id=DATA_REPO, filename="tfidf_matrix_sparse.npz",
                    repo_type="dataset", local_dir=".")

df = pd.read_parquet("papers.parquet")
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
tfidf_matrix = sp.load_npz("tfidf_matrix_sparse.npz")
st_model = SentenceTransformer("all-MiniLM-L6-v2")

@app.get("/")
def root():
    return {"message": "Paper Recommendation API", "docs": "/docs", "health": "/health"}

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


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Paper Recommender</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #2c3e50; }
        input, select, button { padding: 10px; margin: 5px; font-size: 16px; }
        button { background: #3498db; color: white; border: none; cursor: pointer; border-radius: 5px; }
        button:hover { background: #2980b9; }
        .card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; }
        .rank { font-size: 24px; font-weight: bold; color: #3498db; }
        .title { font-size: 16px; font-weight: bold; margin: 5px 0; }
        .meta { color: #666; font-size: 14px; }
        .score { color: #27ae60; font-weight: bold; }
        #query-info { background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 15px 0; }
        #error { color: red; }
    </style>
</head>
<body>
    <h1>🔬 Scientific Paper Recommender</h1>
    <p>Enter a paper ID (0–49999) to find similar arXiv papers.</p>

    <div>
        <input type="number" id="paper-id" placeholder="Paper ID (e.g. 100)" min="0" max="49999" value="100"/>
        <select id="method">
            <option value="hybrid">Hybrid (Best)</option>
            <option value="tfidf">TF-IDF (Fast)</option>
        </select>
        <input type="number" id="k" placeholder="Results" value="5" min="1" max="20"/>
        <button onclick="recommend()">Find Similar Papers</button>
    </div>

    <div id="query-info" style="display:none"></div>
    <div id="error"></div>
    <div id="results"></div>

    <script>
        async function recommend() {
            const idx = document.getElementById("paper-id").value;
            const method = document.getElementById("method").value;
            const k = document.getElementById("k").value;

            document.getElementById("results").innerHTML = "Loading...";
            document.getElementById("error").innerHTML = "";
            document.getElementById("query-info").style.display = "none";

            try {
                const res = await fetch(`/recommend/${idx}?method=${method}&k=${k}`);
                if (!res.ok) throw new Error("Paper not found");
                const data = await res.json();

                document.getElementById("query-info").style.display = "block";
                document.getElementById("query-info").innerHTML = `
                    <strong>Query Paper #${data.query.idx}</strong><br>
                    📄 ${data.query.title}<br>
                    🏷️ Category: ${data.query.category}
                `;

                document.getElementById("results").innerHTML = data.recommendations.map(r => `
                    <div class="card">
                        <span class="rank">#${r.rank}</span>
                        <div class="title">${r.title}</div>
                        <div class="meta">
                            🏷️ ${r.category} &nbsp;|&nbsp;
                            <span class="score">Score: ${r.score}</span>
                        </div>
                    </div>
                `).join("");
            } catch(e) {
                document.getElementById("error").innerHTML = "Error: " + e.message;
                document.getElementById("results").innerHTML = "";
            }
        }
    </script>
</body>
</html>
"""
