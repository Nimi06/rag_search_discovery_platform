import streamlit as st
from rag.rag_pipeline import RAGPipeline

st.set_page_config(page_title="RAG Search & Discovery", layout="wide")
st.title("RAG Search & Discovery Platform")
st.write("Semantic search, hybrid retrieval, re-ranking, and RAG over enterprise documents.")

@st.cache_resource
def load_pipeline():
    return RAGPipeline()

pipeline = load_pipeline()
query = st.text_input("Enter a search question", "How do we monitor model drift?")
top_k = st.slider("Top K", 1, 10, 5)

if st.button("Search"):
    results = pipeline.search(query, top_k=top_k)
    for i, result in enumerate(results, start=1):
        st.subheader(f"Result {i}: {result['doc_id']}")
        st.caption(f"Score: {result.get('rerank_score', 0):.4f} | Source: {result['source']}")
        st.write(result["text"])

if st.button("Generate RAG Answer"):
    output = pipeline.answer(query, top_k=top_k)
    st.subheader("Answer")
    st.write(output["answer"])
