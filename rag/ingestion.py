from pathlib import Path
from typing import List, Dict


def load_documents(directory: str) -> List[Dict[str, str]]:
    """Load .txt and .md documents from a directory."""
    docs = []
    base = Path(directory)
    for path in sorted(base.glob("**/*")):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        docs.append({
            "doc_id": path.stem,
            "source": str(path),
            "text": path.read_text(encoding="utf-8")
        })
    return docs
