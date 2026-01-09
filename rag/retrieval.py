from pathlib import Path
from typing import List, Dict
import json
import numpy as np

from sentence_transformers import SentenceTransformer

def load_embeddings(embeddings_path: Path) -> List[Dict]:
	"""
	Load embedded chunks from disk
	"""
	with embeddings_path.open("r", encoding="utf-8") as f:
		return json.load(f)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
	"""
	Compute cosine similarity between two vectors
	"""
	return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query: str, embedded_chunks: List[Dict], model_name: str="all-MiniLM-L6-v2", top_k: int=3) -> List[Dict]:
	"""
	Retrieve top-k most relevant chunks for a query.
	"""
	model = SentenceTransformer(model_name)
	query_embedding = model.encode(query)

	scored_chunks = []

	for chunk in embedded_chunks:
		chunk_embedding = np.array(chunk["embedding"])
		score = cosine_similarity(query_embedding, chunk_embedding)

		scored_chunks.append({
			"text": chunk["text"],
			"source": chunk["source"],
			"chunk_id": chunk["chunk_id"],
			"score": score	
		})

	scored_chunks.sort(key=lambda x: x["score"], reverse=True)

	return scored_chunks[:top_k]
