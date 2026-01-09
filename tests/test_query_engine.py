# tests/test_query_engine.py

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.query_engine import query_engine


def main():
    query = "What is chunking and why is it important in RAG?"
    answer = query_engine(query)

    print("\nQuery:", query)
    print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()