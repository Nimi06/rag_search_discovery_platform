from rag.ingestion import load_documents
from rag.chunking import chunk_documents
from rag.embeddings import EmbeddingModel
from rag.vector_store import InMemoryVectorStore
from rag.keyword_search import KeywordSearch
from rag.hybrid_search import HybridSearch
from rag.reranker import SimpleReranker


class RAGPipeline:
    def __init__(self, docs_path: str = "data/sample_docs"):
        docs = load_documents(docs_path)
        self.chunks = chunk_documents(docs)
        self.embedding_model = EmbeddingModel()
        embeddings = self.embedding_model.encode([c["text"] for c in self.chunks])
        self.vector_store = InMemoryVectorStore()
        self.vector_store.build(self.chunks, embeddings)
        self.keyword_search = KeywordSearch()
        self.keyword_search.build(self.chunks)
        self.hybrid = HybridSearch(self.vector_store, self.keyword_search, self.embedding_model)
        self.reranker = SimpleReranker()

    def search(self, query: str, top_k: int = 5):
        candidates = self.hybrid.search(query, top_k=top_k * 2)
        return self.reranker.rerank(query, candidates, top_k=top_k)

    def answer(self, query: str, top_k: int = 4) -> dict:
        results = self.search(query, top_k=top_k)
        context = "\n\n".join([r["text"] for r in results])
        # In production, send this context to an LLM. This local answer keeps the repo runnable without API keys.
        answer = (
            "Based on the retrieved enterprise knowledge base, the most relevant information is:\n\n"
            + context[:1200]
        )
        return {"query": query, "answer": answer, "sources": results}
