"""Model factory for Ollama Cloud. Every other module gets its models from here."""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

AGENT_MODEL = os.getenv("AGENT_MODEL", "gemma4:31b")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4:31b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")


def get_llm(model: str = AGENT_MODEL, temperature: float = 0.0, **kwargs) -> ChatOllama:
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        raise RuntimeError("OLLAMA_API_KEY is not set (expected in .env)")
    return ChatOllama(
        model=model,
        base_url=OLLAMA_HOST,
        temperature=temperature,
        client_kwargs={"headers": {"Authorization": f"Bearer {key}"}, "timeout": 120},
        **kwargs,
    )
