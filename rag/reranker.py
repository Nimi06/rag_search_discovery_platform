from typing import List, Dict


class SimpleReranker:
    """Lightweight reranker placeholder.

    In production, replace with a cross-encoder reranker such as
    cross-encoder/ms-marco-MiniLM-L-6-v2.
    """
    def rerank(self, query: str, results: List[Dict], top_k: int = 5) -> List[Dict]:
        query_terms = set(query.lower().split())
        for result in results:
            text_terms = set(result["text"].lower().split())
            overlap = len(query_terms & text_terms)
            result["rerank_score"] = result.get("hybrid_score", 0.0) + 0.03 * overlap
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_k]
