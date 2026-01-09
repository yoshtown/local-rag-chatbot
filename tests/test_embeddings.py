# tests/test_embeddings.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.embeddings import (
    load_chunks,
    embed_chunks,
    save_embeddings
)


def main():
    chunks_path = Path("data/processed/chunks.json")
    output_path = Path("data/processed/embeddings.json")

    chunks = load_chunks(chunks_path)
    embedded_chunks = embed_chunks(chunks)

    save_embeddings(embedded_chunks, output_path)

    print(f"Embedded {len(embedded_chunks)} chunks")
    print("Embedding vector length:", len(embedded_chunks[0]["embedding"]))
    print("Sample metadata:")
    print({
        "source": embedded_chunks[0]["source"],
        "chunk_id": embedded_chunks[0]["chunk_id"]
    })


if __name__ == "__main__":
    main()