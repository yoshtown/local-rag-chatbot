# 2. Convert each chunk into a vector embedding.

from pathlib import Path
from typing import List, Dict
import json

from sentence_transformers import SentenceTransformer

def load_chunks(chunks_path: Path) -> List[Dict]:
	"""
	Load chunked documents from JSON.
	"""
	with chunks_path.open("r", encoding="utf-8") as f:
		return json.load(f)

def embed_chunks(chunks: List[Dict], model_name: str = "all-MiniLM-L6-v2") -> List[Dict]:
	"""
	Generate embeddings for each chunk and attach them to metadata.
	"""
	model = SentenceTransformer(model_name)

	texts = [chunk["text"] for chunk in chunks]
	embeddings = model.encode(texts, show_progress_bar=True)

	embedded_chunks = []

	for chunk, vector in zip(chunks, embeddings):
		embedded_chunks.append({
			"text": chunk["text"],
			"source": chunk["source"],
			"chunk_id": chunk["chunk_id"],
			"embedding": vector.tolist()
		})

	return embedded_chunks

def save_embeddings(embedded_chunks: List[Dict], output_path: Path) -> None:
	"""
	Persist embedded chunks to disk.
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8") as f:
		json.dump(embedded_chunks, f, indent=2)