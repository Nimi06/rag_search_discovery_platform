import pandas as pd
from rag.rag_pipeline import RAGPipeline
from rag.metrics import precision_at_k, recall_at_k, mrr, ndcg_at_k


def run_eval(k: int = 5):
    pipeline = RAGPipeline()
    qrels = pd.read_csv("data/relevance_judgments.csv")
    rows = []
    for query, group in qrels.groupby("query"):
        relevant = set(group[group["relevant"] == 1]["doc_id"].tolist())
        results = pipeline.search(query, top_k=k)
        retrieved = [r["doc_id"] for r in results]
        rows.append({
            "query": query,
            f"precision@{k}": precision_at_k(retrieved, relevant, k),
            f"recall@{k}": recall_at_k(retrieved, relevant, k),
            "mrr": mrr(retrieved, relevant),
            f"ndcg@{k}": ndcg_at_k(retrieved, relevant, k),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    print("\nAverage metrics:")
    print(df.drop(columns=["query"]).mean().to_string())


if __name__ == "__main__":
    run_eval()
