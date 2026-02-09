# scripts/build_index.py

from pathlib import Path

from rag.chunking import load_text_files, chunk_documents, save_chunks
from rag.embeddings import load_chunks, embed_chunks, save_embeddings

# ---- Paths ----
RAW_DATA_DIR = Path("data/raw")
CHUNKS_PATH = Path("data/processed/chunks.json")
EMBEDDINGS_PATH = Path("data/processed/embeddings.json")


def main():
    print("Loading raw documents...")
    documents = load_text_files(RAW_DATA_DIR)

    print("Chunking documents...")
    chunks = chunk_documents(
        documents,
        chunk_size=500,
        overlap=100
    )

    print("Saving chunks...")
    save_chunks(chunks, CHUNKS_PATH)

    print("Loading chunks for embedding...")
    loaded_chunks = load_chunks(CHUNKS_PATH)

    print("Generating embeddings...")
    embedded_chunks = embed_chunks(loaded_chunks)

    print("Saving embeddings...")
    save_embeddings(embedded_chunks, EMBEDDINGS_PATH)

    print("Index build complete!")
    print(f"Chunks saved to: {CHUNKS_PATH}")
    print(f"Embeddings saved to: {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    main()
