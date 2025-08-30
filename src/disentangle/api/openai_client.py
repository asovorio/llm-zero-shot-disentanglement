from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

class OpenAIClient:
    """
    Thin wrapper for OpenAI Chat Completions with optional JSON response_format.
    """
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 400, structured: bool = True):
        if OpenAI is None:
            raise ImportError("openai package not installed. pip install openai>=1.40.0")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured = structured

    def chat(self, system: str, user: str) -> str:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            #temperature=self.temperature, # not used for gpt5-mini
            #max_tokens=self.max_tokens,
        )
        if self.structured:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
