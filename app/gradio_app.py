# app/gradio_app_multiturn.py

import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path so rag can be imported
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import gradio as gr
from rag.query_engine import query_engine

# Optional tweaks
MODEL_NAME = "llama3"
TOP_K = 3


def chat(
    history: List[Tuple[str, str]],
    user_input: str
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Multi-turn chat function.
    history: list of previous messages [(user, bot), ...]
    user_input: current query
    Returns formatted chat text + updated history
    """
    history = history or []

    history.append(("User", user_input))

    answer = query_engine(user_input, top_k=TOP_K, model=MODEL_NAME)

    history.append(("Bot", answer))

    formatted_history = "\n\n".join(f"{role}: {msg}" for role, msg in history)

    return formatted_history, history


def launch_app():
    with gr.Blocks() as demo:
        gr.Markdown(
            "## Local RAG Chatbot (Multi-turn) \n"
            "Ask questions and get grounded answers from your documents!"
        )

        chat_history = gr.State([])

        with gr.Row():
            user_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here..."
            )
            submit_btn = gr.Button("Ask")

        output = gr.Textbox(
            label="Chat History",
            placeholder="Chat will appear here...",
            lines=20
        )

        submit_btn.click(fn=chat, inputs=[chat_history, user_input], outputs=[output, chat_history])
        user_input.submit(fn=chat, inputs=[chat_history, user_input], outputs=[output, chat_history])

    demo.launch()


if __name__ == "__main__":
    launch_app()