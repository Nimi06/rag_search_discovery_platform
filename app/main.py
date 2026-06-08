from fastapi import FastAPI
from pydantic import BaseModel
from rag.rag_pipeline import RAGPipeline

app = FastAPI(title="RAG Search & Discovery Platform", version="1.0.0")
pipeline = RAGPipeline()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/search")
def search(request: QueryRequest):
    return {"results": pipeline.search(request.query, request.top_k)}


@app.post("/answer")
def answer(request: QueryRequest):
    return pipeline.answer(request.query, request.top_k)
