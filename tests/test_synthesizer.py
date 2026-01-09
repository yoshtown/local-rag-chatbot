# tests/test_synthesizer.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.retrieval import load_embeddings, retrieve
from rag.synthesizer import build_prompt, generate_response


def main():
    embeddings_path = Path("data/processed/embeddings.json")

    embedded_chunks = load_embeddings(embeddings_path)

    query = "What is chunking and why is it important in RAG?"

    retrieved = retrieve(
        query=query,
        embedded_chunks=embedded_chunks,
        top_k=3
    )

    prompt = build_prompt(query, retrieved)
    response = generate_response(prompt)

    print("\n--- PROMPT ---\n")
    print(prompt)

    print("\n--- RESPONSE ---\n")
    print(response)


if __name__ == "__main__":
    main()