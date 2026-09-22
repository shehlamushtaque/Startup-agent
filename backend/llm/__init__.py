"""LLM client module - supports Ollama, Groq, DeepSeek, and any OpenAI-compatible endpoint."""

from __future__ import annotations

import os
from typing import Optional

from backend.llm.client import DeepSeekClient, GroqClient, OpenAICompatibleClient

try:
    from backend.llm.ollama_client import OllamaClient
except ImportError:
    OllamaClient = None


def _ollama_available() -> bool:
    if not OllamaClient:
        return False
    try:
        return OllamaClient()._check_availability()
    except Exception:
        return False


def get_llm_client():
    """Get the appropriate LLM client based on environment configuration.

    Priority:
    1. Ollama (local, free) if a server is reachable
    2. Groq (GROQ_API_KEY)
    3. Any OpenAI-compatible endpoint (LLM_API_KEY + LLM_BASE_URL)
    4. DeepSeek (DEEPSEEK_API_KEY)

    Returns None if none are available.
    """
    if _ollama_available():
        try:
            return OllamaClient()
        except Exception:
            pass

    if os.getenv("GROQ_API_KEY"):
        try:
            return GroqClient()
        except Exception:
            pass

    if os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL"):
        try:
            return OpenAICompatibleClient()
        except Exception:
            pass

    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            return DeepSeekClient()
        except Exception:
            pass

    return None


def llm_available() -> bool:
    """True if any LLM provider is configured and reachable."""
    return bool(
        _ollama_available()
        or os.getenv("GROQ_API_KEY")
        or (os.getenv("LLM_API_KEY") and os.getenv("LLM_BASE_URL"))
        or os.getenv("DEEPSEEK_API_KEY")
    )
