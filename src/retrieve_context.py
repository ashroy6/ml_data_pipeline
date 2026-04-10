cat > src/retrieve_context.py <<'EOF'
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


PROJECT_ROOT = Path(__file__).resolve().parent.parent
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

COLLECTION_NAME = "project1_rag"
EMBED_MODEL = "nomic-embed-text"


def retrieve_relevant_context(query: str, n_results: int = 4) -> list[dict]:
    client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    embed_fn = embedding_functions.OllamaEmbeddingFunction(
        url="http://127.0.0.1:11434/api/embeddings",
        model_name=EMBED_MODEL,
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    result = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    items = []
    for doc, meta in zip(documents, metadatas):
        meta = meta or {}
        items.append(
            {
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", -1),
                "text": doc,
            }
        )

    return items


def retrieve_context(query: str, top_k: int = 4) -> list[dict]:
    return retrieve_relevant_context(query=query, n_results=top_k)
EOF