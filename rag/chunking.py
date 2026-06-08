from typing import List, Dict


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80) -> List[str]:
    """Simple word-based chunking with overlap."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_documents(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    records = []
    for doc in documents:
        for idx, chunk in enumerate(chunk_text(doc["text"])):
            records.append({
                "chunk_id": f"{doc['doc_id']}::chunk_{idx}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "text": chunk
            })
    return records
