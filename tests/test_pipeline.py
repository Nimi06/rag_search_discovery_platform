from rag.rag_pipeline import RAGPipeline


def test_pipeline_search_returns_results():
    pipeline = RAGPipeline(docs_path="data/sample_docs")
    results = pipeline.search("model drift monitoring", top_k=3)
    assert len(results) > 0
    assert "text" in results[0]
