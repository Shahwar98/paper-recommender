
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

@app.get("/", response_class=HTMLResponse)
def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")

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

@app.get("/search")
def search_by_text(q: str, k: int = 10, method: str = "hybrid"):
    """Search for papers by free text query"""
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Transform query using TF-IDF
    query_vec = vectorizer.transform([q.lower().strip()])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_50 = np.argsort(scores)[::-1][:50]
    
    if method == "tfidf":
        return {
            "query": q,
            "method": "tfidf",
            "recommendations": [
                {"rank": i+1, "title": df.loc[int(idx), "title"],
                 "category": df.loc[int(idx), "primary_category"],
                 "score": round(float(scores[idx]), 4)}
                for i, idx in enumerate(top_50[:k])
            ]
        }
    
    # Semantic reranking
    cand_texts = [df.loc[int(i), "combined_text"] for i in top_50]
    embeddings = st_model.encode([q] + cand_texts,
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
            "arxiv_id": df.loc[int(idx), "id"],
            "score": round(0.3 * float(scores[idx]) + 0.7 * sem_scores[i], 4)
        })
    results.sort(key=lambda x: x["score"], reverse=True)
    for i, r in enumerate(results[:k]):
        r["rank"] = i + 1
    
    return {"query": q, "method": method, "recommendations": results[:k]}


@app.get("/ui", response_class=HTMLResponse)
async def ui():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Scientific Paper Recommender</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #f5f7fa; }
        .header { background: #2c3e50; color: white; padding: 30px 20px; text-align: center; }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.8; font-size: 15px; }
        .container { max-width: 800px; margin: 30px auto; padding: 0 20px; }
        .search-box { background: white; padding: 25px; border-radius: 12px; 
                      box-shadow: 0 2px 10px rgba(0,0,0,0.08); margin-bottom: 20px; }
        .search-row { display: flex; gap: 10px; }
        input[type=text] { flex: 1; padding: 12px 16px; font-size: 16px; 
                           border: 2px solid #e0e0e0; border-radius: 8px; outline: none; }
        input[type=text]:focus { border-color: #3498db; }
        button { background: #3498db; color: white; border: none; padding: 12px 24px; 
                 font-size: 16px; border-radius: 8px; cursor: pointer; white-space: nowrap; }
        button:hover { background: #2980b9; }
        .examples { margin-top: 12px; font-size: 13px; color: #666; }
        .examples span { color: #3498db; cursor: pointer; margin-right: 12px; }
        .examples span:hover { text-decoration: underline; }
        .card { background: white; padding: 20px; margin-bottom: 12px; border-radius: 10px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.06); display: flex; gap: 15px; }
        .rank { font-size: 28px; font-weight: bold; color: #3498db; min-width: 40px; }
        .content .title { font-size: 16px; font-weight: 600; color: #2c3e50; margin-bottom: 6px; }
        .content .meta { font-size: 13px; color: #888; }
        .score { color: #27ae60; font-weight: bold; }
        .loading { text-align: center; padding: 40px; color: #666; font-size: 18px; }
        .query-info { background: #eaf4fb; border-left: 4px solid #3498db; 
                      padding: 12px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Scientific Paper Recommender</h1>
        <p>Search across 50,000 arXiv papers using AI-powered semantic search</p>
    </div>
    <div class="container">
        <div class="search-box">
            <div class="search-row">
                <input type="text" id="query" placeholder="e.g. deep learning for protein folding" 
                       onkeypress="if(event.key==='Enter') search()"/>
                <button onclick="search()">Search</button>
            </div>
            <div class="examples">
                Try: 
                <span onclick="setQuery('transformer attention mechanism NLP')">transformers NLP</span>
                <span onclick="setQuery('reinforcement learning robotics')">RL robotics</span>
                <span onclick="setQuery('graph neural networks molecules')">graph neural nets</span>
                <span onclick="setQuery('diffusion models image generation')">diffusion models</span>
            </div>
        </div>
        <div id="results"></div>
    </div>

    <script>
        function setQuery(text) {
            document.getElementById("query").value = text;
            search();
        }

        async function search() {
            const q = document.getElementById("query").value.trim();
            if (!q) return;
            
            document.getElementById("results").innerHTML = 
                '<div class="loading">🔍 Searching 50,000 papers...</div>';

            try {
                const res = await fetch(`/search?q=${encodeURIComponent(q)}&method=hybrid&k=8`);
                if (!res.ok) throw new Error("Search failed");
                const data = await res.json();

                document.getElementById("results").innerHTML = `
                    <div class="query-info">
                        Showing top results for: <strong>"${data.query}"</strong> 
                        &nbsp;·&nbsp; ${data.recommendations.length} papers found
                    </div>
                    ${data.recommendations.map(r => `
                        <div class="card">
                            <div class="rank">#${r.rank}</div>
                            <div class="content">
                                <div class="title">
                                    <a href="https://arxiv.org/abs/${r.arxiv_id}" target="_blank">
                                        ${r.title}
                                    </a>
                                </div>
                                <div class="meta">
                                    🏷️ ${r.category} &nbsp;|&nbsp;
                                    <span class="score">Score: ${r.score}</span>
                                </div>
                            </div>
                        </div>
                    `).join("")}
                `;
            } catch(e) {
                document.getElementById("results").innerHTML = 
                    `<div style="color:red;padding:20px">Error: ${e.message}</div>`;
            }
        }
    </script>
</body>
</html>
"""
