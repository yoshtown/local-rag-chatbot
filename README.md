# Local RAG Chatbot (From Scratch, Fully Offline)

A **local, end-to-end Retrieval-Augmented Generation (RAG) chatbot** built in Python.
This project demonstrates how modern RAG systems actually work under the hood—without relying on managed services or opaque frameworks.

Everything runs **locally**, including embeddings, retrieval, and LLM inference via **Ollama**.

---

## Why This Project Exists

Most RAG tutorials skip fundamentals or hide complexity behind libraries.
This project intentionally builds each layer explicitly so you can:

* Understand how document chunking affects retrieval quality
* See how embeddings power semantic search
* Control hallucination behavior through prompt discipline
* Debug RAG failures at the system level

This is an **applied ML systems project**, not just an LLM demo.

---

## What the System Does

1. Loads raw text documents
2. Splits them into overlapping chunks
3. Converts chunks into vector embeddings
4. Stores embeddings locally
5. Retrieves top-K relevant chunks per query
6. Injects retrieved context into a constrained prompt
7. Generates answers using a **local LLM**
8. Refuses to answer when information is missing

---

## Architecture Overview

```
User Query
   ↓
Retriever (cosine similarity)
   ↓
Top-K Chunks
   ↓
Prompt Assembler
   ↓
Local LLM (Ollama)
   ↓
Grounded Answer
```

---

## Project Structure

```
local-rag-chatbot/
├── rag/
│   ├── chunking.py        # Document loading + chunking
│   ├── embeddings.py     # Vector embedding generation
│   ├── retrieval.py      # Similarity search logic
│   ├── synthesizer.py    # LLM prompt + generation
│   └── query_engine.py   # End-to-end RAG pipeline
│
├── app/
│   ├── gradio_app.py
│   └── gradio_app_multiturn.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── tests/
│
├── requirements.txt
└── README.md
```

---

## Setup (Windows)

### Create and activate a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
pip install -r requirements.txt
```

---

## Local LLM Setup (Ollama)

Install Ollama from:

```
https://ollama.com
```

Restart your terminal, then pull a model:

```powershell
ollama run llama3
```

---

## Running the Pipeline

### Chunk documents

```powershell
python rag/chunking.py
```

### Generate embeddings

```powershell
python rag/embeddings.py
```

### Launch the chatbot

```powershell
python app/gradio_app_multiturn.py
```

Open the local Gradio URL in your browser.

---

## Example Query

```
What is chunking and why is it important in RAG?
```

* Answer is grounded in retrieved documents
* No hallucination
* Explicit refusal if context is missing

---

## Tech Stack

* Python
* Sentence Transformers
* Scikit-learn
* Ollama (local LLM inference)
* Gradio (UI)

---

## Key Takeaways

This project demonstrates:

* Practical RAG system design
* Embedding-based retrieval
* Prompt-level hallucination control
* Local, privacy-preserving LLM usage
* Modular ML system architecture

This is the kind of project that scales naturally into **production RAG systems**.

---

## Future Extensions

* FAISS or Chroma vector stores
* Token-based chunking
* Source citation display
* PDF ingestion
* Streaming responses
* Multi-document collections
