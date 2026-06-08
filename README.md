# RAG Search & Discovery Platform

A production-style **Retrieval-Augmented Generation (RAG), semantic search, and search relevance evaluation project** built for enterprise knowledge discovery use cases.

This project demonstrates skills relevant to roles in **Search Relevance, NLP, Generative AI, ML Engineering, MLOps, and Retrieval Systems**.

## Why this project matters

Modern search and discovery systems combine:

- Query understanding
- Document ingestion and chunking
- Embedding-based semantic retrieval
- Keyword search
- Hybrid retrieval
- Re-ranking
- Retrieval-Augmented Generation
- Offline relevance evaluation
- API serving
- Monitoring-ready architecture

This repository implements those concepts in a clean, modular, recruiter-friendly way.

## Key Features

- Document ingestion pipeline for `.txt` and `.md` files
- Text chunking with metadata preservation
- TF-IDF keyword retrieval baseline
- Embedding-based semantic retrieval using `sentence-transformers`
- Hybrid search combining semantic and keyword signals
- Cross-encoder re-ranking support
- RAG answer generation interface
- Offline search relevance evaluation using Precision@K, Recall@K, MRR, and NDCG@K
- FastAPI service for search and question answering
- Streamlit demo UI
- Unit tests
- Dockerfile
- GitHub Actions CI workflow

## Architecture

```text
Documents
   ↓
Ingestion + Chunking
   ↓
Embedding Generation + Keyword Index
   ↓
Hybrid Retrieval
   ↓
Re-ranking
   ↓
RAG Answer Generation
   ↓
API / UI / Evaluation
```

## Tech Stack

- Python
- FastAPI
- Streamlit
- SentenceTransformers
- scikit-learn
- NumPy / Pandas
- Pytest
- Docker
- GitHub Actions

## Project Structure

```text
rag-search-discovery-platform/
├── app/
│   ├── main.py                 # FastAPI application
│   └── streamlit_app.py        # Demo UI
├── rag/
│   ├── chunking.py             # Text chunking
│   ├── ingestion.py            # Document loading
│   ├── embeddings.py           # Embedding model wrapper
│   ├── keyword_search.py       # TF-IDF/BM25-style keyword baseline
│   ├── vector_store.py         # In-memory vector search
│   ├── hybrid_search.py        # Hybrid retrieval logic
│   ├── reranker.py             # Re-ranking interface
│   ├── rag_pipeline.py         # End-to-end RAG pipeline
│   └── metrics.py              # Relevance metrics
├── evals/
│   └── relevance_eval.py       # Offline evaluation script
├── data/
│   ├── sample_docs/            # Sample enterprise documents
│   └── relevance_judgments.csv # Example qrels
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

### 4. Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

### 5. Run evaluation

```bash
python evals/relevance_eval.py
```

### 6. Run tests

```bash
pytest
```

## Example Queries

Try these queries:

- `How do we monitor model drift?`
- `What is the process for incident response?`
- `How should customer data be protected?`
- `What are the steps in the ML deployment lifecycle?`

## Relevance Metrics

The evaluation module supports:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- NDCG@K

These metrics are commonly used for search relevance and ranking evaluation.

## Resume Talking Points

You can describe this project as:

> Built a production-style RAG and hybrid search platform combining semantic vector retrieval, keyword search, query understanding, re-ranking, and offline relevance evaluation using Precision@K, Recall@K, MRR, and NDCG@K. Exposed the system through FastAPI and Streamlit with CI tests and Dockerized deployment.

## Future Enhancements

- Add FAISS or Chroma persistent vector store
- Add OpenSearch / Elasticsearch keyword backend
- Add LLM-based query rewriting
- Add MLflow tracking for retrieval experiments
- Add online A/B testing simulation
- Add Databricks notebook version using Spark and Delta tables

## Author

Nimika Aggarwal  
GitHub: [Nimi06](https://github.com/Nimi06)
