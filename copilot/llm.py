"""Gemini access with automatic fallback.

During the hackathon the full Flash model often returned 503 "high demand", and the
free tier allows only ~20 requests per day *per model*. So every call walks a chain
of models (GEMINI_MODEL, then each of GEMINI_FALLBACK_MODEL, comma-separated)
instead of failing on the first overloaded or exhausted one.
"""
import os

from langchain_google_genai import ChatGoogleGenerativeAI

from copilot import config


def chat_model(temperature: float = 0.0):
    """Returns (primary, [fallbacks]) chat models."""
    def make(name):
        return ChatGoogleGenerativeAI(model=name, temperature=temperature, max_retries=1, timeout=15,
                                      google_api_key=os.environ.get("GOOGLE_API_KEY"))
    fallbacks = [m.strip() for m in config.GEMINI_FALLBACK_MODEL.split(",") if m.strip()]
    return make(config.GEMINI_MODEL), [make(m) for m in fallbacks]


def with_fallback(primary, fallbacks, tools=None):
    """Bind tools (if any) to every model, then chain them primary -> fallbacks in order."""
    if tools:
        primary, fallbacks = primary.bind_tools(tools), [f.bind_tools(tools) for f in fallbacks]
    return primary.with_fallbacks(fallbacks) if fallbacks else primary
