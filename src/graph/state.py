"""LangGraph agent state definition."""
from __future__ import annotations

import operator
from typing import Annotated, Any, Sequence

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dicts. Right values overwrite left,
    except nested dicts are merged recursively."""
    merged = left.copy()
    for key, value in right.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


class AgentState(TypedDict):
    """State passed between all nodes in the LangGraph workflow.

    messages: Accumulated agent messages (append-only).
    data: Shared data — tickers, portfolio, dates, analyst_signals, current_prices.
    metadata: Config — model_name, model_provider, show_reasoning.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    data: Annotated[dict, merge_dicts]
    metadata: Annotated[dict, merge_dicts]
