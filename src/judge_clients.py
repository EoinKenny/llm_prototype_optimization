"""User-supplied API adapter for the qualitative LLM judges."""
from __future__ import annotations


def call_judge_api(
    prompts: list[str],
    provider: str,
    model_id: str,
) -> list[str]:
    """Send a batch of prompts to one judge model.

    Parameters
    ----------
    prompts:
        Prompt strings to evaluate.
    provider:
        Provider name from ``src.config.JUDGES``.
    model_id:
        Model identifier from ``src.config.JUDGES``.

    Returns
    -------
    list[str]
        One response per prompt, in the same order as ``prompts``.

    Replace the body of this function with your own API code. The rest of the
    pipeline is provider-independent.
    """
    raise NotImplementedError(
        "Implement call_judge_api() in src/judge_clients.py using your API client."
    )
