# src/api/llm_client.py

"""
Shared OpenAI LLM client helpers.

All code that talks to OpenAI should go through this module so
the rest of the project doesn't need to worry about API keys,
client setup, etc.
"""

import os
import json

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """
    Lazily create and cache a single OpenAI client.

    OPENAI_API_KEY is read from environment (or .env via load_dotenv()).
    """
    global _client
    if _client is None:
        # The OpenAI client will read OPENAI_API_KEY from env.
        _client = OpenAI()
    return _client


def chat_completion(model: str, system_prompt: str, user_prompt: str, max_tokens: int = 600) -> str:
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content
