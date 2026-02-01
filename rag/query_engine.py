from pathlib import Path
from typing import List, Dict

from rag.retrieval import load_embeddings, retrieve
from rag.synthesizer import build_prompt, generate_response

# Paths to stored embeddings
EMBEDDINGS_PATH = Path("data/processed/embeddings.json")

# Load embeddings once at a module load
embedded_chunks: List[Dict] = load_embeddings(EMBEDDINGS_PATH)

def query_engine(query: str, top_k: int=3, model: str="llama3") -> str:
	"""
	Full RAG query engine: retrieves relevant chunks and generates
	a grounded answer from the local LLM.
	"""
	# Step 1: Retrieve top-k relevant chunks
	retrieved_chunks = retrieve(query=query, embedded_chunks=embedded_chunks, top_k=top_k)

	# Step 2: Build prompt
	prompt = build_prompt(query, retrieved_chunks)

	# Step 3: Generate response
	answer = generate_response(prompt, model=model)

	return answer