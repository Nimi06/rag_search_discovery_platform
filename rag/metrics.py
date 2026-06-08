import math
from typing import List, Set


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    retrieved_k = retrieved[:k]
    return sum(1 for doc in retrieved_k if doc in relevant) / max(k, 1)


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    return sum(1 for doc in retrieved_k if doc in relevant) / len(relevant)


def mrr(retrieved: List[str], relevant: Set[str]) -> float:
    for idx, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    dcg = 0.0
    for idx, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            dcg += 1.0 / math.log2(idx + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0
