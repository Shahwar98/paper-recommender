---
title: Paper Recommender
emoji: 📄
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Scientific Paper Recommendation System

An end-to-end ML system that recommends arXiv papers using two-stage retrieval.

## API Endpoints

- `GET /health` - Health check
- `GET /recommend/{paper_idx}?k=10&method=hybrid` - Get recommendations
- `GET /paper/{paper_idx}` - Get paper details
- `GET /stats` - Dataset statistics
- `GET /docs` - Interactive Swagger UI
