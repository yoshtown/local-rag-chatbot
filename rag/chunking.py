# 1. Break each document into chunks of roughly 300-500 characters.

from pathlib import Path
from typing import List, Dict
import json

def load_text_files(data_dir: Path) -> List[Dict]:
	"""
	Load all *.txt files from a directory and return their contents
	along with basic metadata.
	"""
	documents = []

	for file_path in data_dir.glob("*.txt"):
		text = file_path.read_text(encoding="utf-8")

		documents.append({
				"text": text,
				"source": file_path.name
			})

	return documents

def chunk_text(text: str, chunk_size: int=500, overlap: int=50) -> List[str]:
	"""
	Split text into overlapping character-based chunks
	"""
	chunks = []
	start = 0

	while start < len(text):
		end = start + chunk_size
		chunk = text[start:end]
		chunks.append(chunk)
		start = end - overlap

	return chunks

def chunk_documents(documents: List[Dict]) -> List[Dict]:
	"""
	Chunk all documents and attach metadata to each chunk.
	"""
	all_chunks = []

	for doc in documents:
		text_chunks = chunk_text(doc["text"])

		for i, chunk in enumerate(text_chunks):
			all_chunks.append({
				"text": chunk,
				"source": doc["source"],
				"chunk_id": i	
			})

	return all_chunks

def save_chunks(chunks: List[Dict], output_path: Path) -> None:
	"""
	Save chunked documents to a JSON file
	"""
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", encoding="utf-8") as f:
		json.dump(chunks, f, indent=2, ensure_ascii=False)

