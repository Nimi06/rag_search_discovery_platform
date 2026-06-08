from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class KeywordSearch:
    def __init__(self):
        self.records: List[Dict[str, str]] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = None

    def build(self, records: List[Dict[str, str]]) -> None:
        self.records = records
        texts = [r["text"] for r in records]
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.matrix is None or not self.records:
            return []
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().reshape(-1)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [{**self.records[i], "keyword_score": float(scores[i])} for i in idxs]
