from typing import List, Dict


class HybridSearch:
    def __init__(self, vector_store, keyword_search, embedding_model, alpha: float = 0.65):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.embedding_model = embedding_model
        self.alpha = alpha

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_emb = self.embedding_model.encode([query])[0]
        semantic = self.vector_store.search(query_emb, top_k=top_k * 3)
        keyword = self.keyword_search.search(query, top_k=top_k * 3)

        merged = {}
        for item in semantic:
            merged.setdefault(item["chunk_id"], item)
            merged[item["chunk_id"]]["semantic_score"] = item.get("semantic_score", 0.0)
        for item in keyword:
            merged.setdefault(item["chunk_id"], item)
            merged[item["chunk_id"]]["keyword_score"] = item.get("keyword_score", 0.0)

        results = []
        for item in merged.values():
            s = item.get("semantic_score", 0.0)
            k = item.get("keyword_score", 0.0)
            item["hybrid_score"] = self.alpha * s + (1 - self.alpha) * k
            results.append(item)
        return sorted(results, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]
