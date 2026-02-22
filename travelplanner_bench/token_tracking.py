"""Unified token tracking and cost estimation across frameworks."""

from __future__ import annotations

from travelplanner_bench.backend import TokenUsage
from travelplanner_bench.models import IterationLog

# Model pricing (USD per 1M tokens)
MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
    "claude-opus-4-6": {"input": 15.00, "output": 75.00},
    # Fireworks (OSS models — priced by param tier)
    "gpt-oss-120b": {"input": 0.90, "output": 0.90},
    "gpt-oss-20b": {"input": 0.20, "output": 0.20},
    "deepseek-v3p2": {"input": 0.90, "output": 0.90},
    "kimi-k2p5": {"input": 0.90, "output": 0.90},
    "mixtral-8x22b-instruct": {"input": 0.90, "output": 0.90},
    "llama-v3p3-70b-instruct": {"input": 0.90, "output": 0.90},
    "qwen2p5-72b-instruct": {"input": 0.90, "output": 0.90},
    # Groq (free tier / usage-based)
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "qwen3-32b": {"input": 0.34, "output": 0.40},
    "llama-4-scout-17b-16e-instruct": {"input": 0.20, "output": 0.60},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    # Google
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
}


def estimate_cost(usage: TokenUsage, model: str) -> float:
    """Estimate USD cost from token usage and model.

    Falls back to $1.00/1M for unknown models.
    """
    # Normalize model name (strip provider prefix)
    model_short = model.rsplit("/", 1)[-1]
    pricing = MODEL_PRICING.get(model_short, {"input": 1.0, "output": 1.0})
    return (
        usage.input_tokens * pricing["input"] / 1_000_000
        + usage.output_tokens * pricing["output"] / 1_000_000
    )


def extract_opensymbolicai_tokens(iteration_logs: list[IterationLog]) -> TokenUsage:
    """Extract unified TokenUsage from OpenSymbolicAI iteration logs."""
    usage = TokenUsage()
    for il in iteration_logs:
        usage.input_tokens += il.input_tokens
        usage.output_tokens += il.output_tokens
        usage.llm_calls += 1
        usage.time_seconds += il.time_seconds
        if il.phase == "retrieval":
            usage.retrieval_input_tokens += il.input_tokens
            usage.retrieval_output_tokens += il.output_tokens
            usage.retrieval_llm_calls += 1
            usage.retrieval_time_seconds += il.time_seconds
        elif il.phase == "assembly":
            usage.assembly_input_tokens += il.input_tokens
            usage.assembly_output_tokens += il.output_tokens
            usage.assembly_llm_calls += 1
            usage.assembly_time_seconds += il.time_seconds
    usage.compute_totals()
    return usage
