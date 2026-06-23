---
title: Paper Recommender
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Scientific Paper Recommendation System

An end-to-end ML system that finds relevant arXiv papers using a two-stage retrieval pipeline — TF-IDF candidate retrieval followed by semantic reranking with sentence transformers.

## Live Demo

[shahwar98-paper-recommender.hf.space](https://shahwar98-paper-recommender.hf.space)

Type anything like *"transformer attention mechanism"* or *"reinforcement learning robotics"* and get semantically relevant papers instantly.

## Results

| Method | Precision@10 | NDCG@10 |
|--------|-------------|---------|
| TF-IDF Baseline | 0.349 | 0.583 |
| Hybrid (TF-IDF + Semantic) | **0.419** | **0.866** |

Semantic reranking improved Precision@10 by **20.1%** and NDCG@10 by **48.4%** over the lexical baseline.

Two embedding models were evaluated (MiniLM and SPECTER2). MiniLM was selected for equivalent NDCG@10 performance with a 5x smaller footprint (90MB vs 440MB).

## Architecture
User Query

|

v

TF-IDF Retrieval -- Top-50 candidates (fast, lexical)

|

v

Sentence Transformer Reranking -- Top-K results (semantic)

|

v

FastAPI Response

## Tech Stack

| Layer | Tools |
|-------|-------|
| ML | Scikit-learn TF-IDF, HuggingFace all-MiniLM-L6-v2 |
| Data | 2,000+ recent arXiv CS/ML papers via arXiv API |
| Evaluation | Precision@K, NDCG@K, MLflow experiment tracking |
| Serving | FastAPI + Uvicorn |
| Storage | SQLite via SQLAlchemy, HuggingFace Dataset Hub |
| MLOps | Docker, GitHub Actions CI/CD |
| Deployment | HuggingFace Spaces |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web UI |
| `GET /search?q=your+query` | Search by free text |
| `GET /recommend/{paper_idx}` | Recommend by paper index |
| `GET /health` | Health check |
| `GET /stats` | Dataset statistics |
| `GET /docs` | Swagger UI |

## Run Locally

```bash
git clone https://github.com/Shahwar98/paper-recommender
cd paper-recommender
pip install -r requirements.txt
uvicorn main:app --reload
```

## Docker

```bash
docker build -t paper-recommender .
docker run -p 8000:8000 paper-recommender
```

## Author

Shahwar Ahmed Khaleel
[GitHub](https://github.com/Shahwar98) | [LinkedIn](https://www.linkedin.com/in/shahwar-ahmed-khaleel-621524194/)
