# tests/test_chunking.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.chunking import (
    load_text_files,
    chunk_documents,
    save_chunks
)


def main():
    raw_dir = Path("data/raw")
    output_path = Path("data/processed/chunks.json")

    docs = load_text_files(raw_dir)
    chunks = chunk_documents(docs)

    save_chunks(chunks, output_path)

    print(f"Saved {len(chunks)} chunks to {output_path}")
    print("First chunk:")
    print(chunks[0])


if __name__ == "__main__":
    main()