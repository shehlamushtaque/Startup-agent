"""LLM client module - supports DeepSeek and Ollama."""

from __future__ import annotations

import os
from typing import Optional, Union

from backend.llm.client import DeepSeekClient

try:
    from backend.llm.ollama_client import OllamaClient
except ImportError:
    OllamaClient = None


def get_llm_client() -> Optional[Union[DeepSeekClient, "OllamaClient"]]:
    """Get the appropriate LLM client based on environment configuration.
    
    Priority:
    1. Ollama (if OLLAMA_BASE_URL is set or Ollama is available)
    2. DeepSeek (if DEEPSEEK_API_KEY is set)
    
    Returns None if neither is available.
    """
    # Check for Ollama first (local, free)
    if OllamaClient:
        try:
            ollama = OllamaClient()
            # Quick check if server is available
            if ollama._check_availability():
                return ollama
        except Exception:
            pass
    
    # Fall back to DeepSeek
    if os.getenv("DEEPSEEK_API_KEY"):
        try:
            return DeepSeekClient()
        except Exception:
            pass
    
    return None
