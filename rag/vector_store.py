from typing import List, Dict
import numpy as np


class InMemoryVectorStore:
    def __init__(self):
        self.records: List[Dict[str, str]] = []
        self.embeddings: np.ndarray | None = None

    def build(self, records: List[Dict[str, str]], embeddings: np.ndarray) -> None:
        self.records = records
        self.embeddings = embeddings

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        if self.embeddings is None or not self.records:
            return []
        scores = self.embeddings @ query_embedding.reshape(-1)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [{**self.records[i], "semantic_score": float(scores[i])} for i in idxs]
