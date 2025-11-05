from __future__ import annotations
import json
from anthropic import Anthropic


class AnthropicClient:
    """
    Thin wrapper around Anthropic Messages API.
    If structured=True, we force JSON-only output via a single tool.
    """
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 400, structured: bool = True):
        if Anthropic is None:
            raise ImportError("anthropic package not installed. `pip install anthropic`")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured = structured
        self.client = Anthropic()  # reads ANTHROPIC_API_KEY from env

    def chat(self, system: str, user: str) -> str:
        if self.structured:
            # Define a single "emit_json" tool with an open schema so every method can reuse it
            tools = [{
                "name": "emit_json",
                "description": "Return ONLY the final JSON object requested by the user.",
                "input_schema": {"type": "object", "additionalProperties": True}
            }]
            msg = self.client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                tools=tools,
                tool_choice={"type": "tool", "name": "emit_json"},  # force tool output
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            # Grab the tool call payload and return it as a JSON string
            for block in getattr(msg, "content", []):
                if getattr(block, "type", None) == "tool_use" and getattr(block, "name", "") == "emit_json":
                    return json.dumps(getattr(block, "input", {}), ensure_ascii=False)
            # Fallback (shouldn't happen); return any text blocks as-is
            parts = [getattr(b, "text", "") for b in getattr(msg, "content", []) if getattr(b, "type", None) == "text"]
            return "".join(parts)
        else:
            msg = self.client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            parts = [getattr(b, "text", "") for b in getattr(msg, "content", []) if getattr(b, "type", None) == "text"]
            return "".join(parts)
