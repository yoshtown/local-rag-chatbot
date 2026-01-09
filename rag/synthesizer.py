from typing import List, Dict
import subprocess

def build_prompt(query: str, context_chunks: List[Dict]) -> str:
	"""
	Construct a ground prompt using retrieved context.
	"""
	context_text = "\n\n".join(
		f"[Source: {chunk['source']} | Chunk {chunk['chunk_id']}]\n{chunk['text']}"
		for chunk in context_chunks
	)

	prompt = f"""
	You are a helpful assistant answering questions using ONLY the context below.
	If the answer cannot be found in the context, say "I don't know based on the provided documents."

	Context:
	{context_text}

	Question:
	{query}

	Answer:
	""".strip()

	return prompt

def generate_response(prompt: str, model: str="llama3") -> str:
	"""
	Call a local LLM via Ollama to generate a response
	"""
	result = subprocess.run(
		["ollama", "run", model],
		input=prompt,
		text=True,
		capture_output=True
	)

	return result.stdout.strip()