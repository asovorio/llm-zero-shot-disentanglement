from .openai_client import OpenAIClient
from .anthropic_client import AnthropicClient


def get_client(provider: str, *, model: str, temperature: float = 0.0, max_tokens: int = 400, structured: bool = True):
    p = (provider or "openai").lower()
    if p in {"openai", "oai"}:
        return OpenAIClient(model=model, temperature=temperature, max_tokens=max_tokens, structured=structured)
    if p in {"anthropic", "claude"}:
        if AnthropicClient is None:
            raise ImportError("anthropic package not installed. `pip install anthropic`")
        return AnthropicClient(model=model, temperature=temperature, max_tokens=max_tokens, structured=structured)
    raise ValueError(f"Unknown provider: {provider}")


__all__ = ["OpenAIClient", "AnthropicClient", "get_client"]