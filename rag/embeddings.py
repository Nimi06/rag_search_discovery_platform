from typing import List
import numpy as np


class EmbeddingModel:
    """Wrapper around SentenceTransformers with a deterministic fallback.

    The fallback keeps tests and demos usable when model downloads are unavailable.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = None
        self.model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
        except Exception:
            self.model = None

    def encode(self, texts: List[str]) -> np.ndarray:
        if self.model is not None:
            return np.asarray(self.model.encode(texts, normalize_embeddings=True))
        # fallback: hashed bag-of-words vectors
        vectors = []
        dim = 384
        for text in texts:
            vec = np.zeros(dim, dtype=float)
            for token in text.lower().split():
                vec[hash(token) % dim] += 1.0
            norm = np.linalg.norm(vec) or 1.0
            vectors.append(vec / norm)
        return np.vstack(vectors)
