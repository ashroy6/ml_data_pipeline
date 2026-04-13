from pathlib import Path
import os

import chromadb
from chromadb.utils import embedding_functions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "rag" / "knowledge_base"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "project1_rag"
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", "http://ollama:11434/api/embeddings")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks = []
    start = 0
    while start < len(cleaned):
        end = start + chunk_size
        chunks.append(cleaned[start:end])
        if end >= len(cleaned):
            break
        start = end - overlap
    return chunks


def main() -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    KB_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    embed_fn = embedding_functions.OllamaEmbeddingFunction(
        url=OLLAMA_EMBED_URL,
        model_name=EMBED_MODEL,
    )

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"description": "RAG knowledge base for Project 1 + P3 analysis"},
    )

    added = 0

    for file_path in sorted(KB_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        chunks = chunk_text(text)
        if not chunks:
            continue

        ids = [f"{file_path.stem}-{i}" for i in range(len(chunks))]
        metadatas = [{"source": file_path.name, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            ids=ids,
            documents=chunks,
            metadatas=metadatas,
        )
        added += len(chunks)
        print(f"Indexed {file_path.name}: {len(chunks)} chunks")

    print(f"Done. Total chunks indexed: {added}")


if __name__ == "__main__":
    main()
