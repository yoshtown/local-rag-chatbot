# tests/test_retrieval.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.retrieval import load_embeddings, retrieve

def main():
    embeddings_path = Path("data/processed/embeddings.json")

    embedded_chunks = load_embeddings(embeddings_path)

    query = "What is chunking in RAG?"

    results = retrieve(
        query=query,
        embedded_chunks=embedded_chunks,
        top_k=3
    )

    print(f"Query: {query}\n")

    for i, result in enumerate(results, start=1):
        print(f"Result {i}")
        print(f"Source: {result['source']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text preview: {result['text'][:200]}")
        print("-" * 40)


if __name__ == "__main__":
    main()