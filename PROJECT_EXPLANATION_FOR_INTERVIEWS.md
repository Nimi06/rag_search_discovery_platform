# Interview Explanation Guide

## 30-second pitch

This project is a production-style RAG and hybrid search platform for enterprise knowledge discovery. It ingests documents, chunks them, creates semantic embeddings, builds a keyword retrieval baseline, combines both signals using hybrid retrieval, reranks results, and exposes the system through FastAPI and Streamlit. It also includes an offline relevance evaluation framework using Precision@K, Recall@K, MRR, and NDCG@K.

## Why it is relevant to search relevance roles

Search relevance teams care about retrieval quality, ranking quality, experimentation speed, query understanding, and evaluation. This project covers those areas through modular retrieval components and offline metrics.

## How hybrid search works here

The pipeline retrieves candidates using two paths:

1. Semantic search using embeddings and cosine similarity.
2. Keyword search using TF-IDF as a BM25-style baseline.

The scores are combined using a weighted hybrid score and then passed through a reranking layer.

## How to improve it for production

- Replace in-memory vector store with FAISS, Chroma, Milvus, or Pinecone.
- Replace TF-IDF with Elasticsearch/OpenSearch BM25.
- Add cross-encoder reranking.
- Add LLM-based query rewriting.
- Track experiments with MLflow.
- Deploy using Kubernetes.
- Add real user feedback and online A/B tests.
