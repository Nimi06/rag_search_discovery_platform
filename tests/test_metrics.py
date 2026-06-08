from rag.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k


def test_ranking_metrics():
    retrieved = ["a", "b", "c"]
    relevant = {"b", "d"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mrr(retrieved, relevant) == 0.5
    assert ndcg_at_k(retrieved, relevant, 3) > 0
