"""Agent registry for LangGraph workflow configuration."""
from __future__ import annotations

from typing import Any, Callable

from src.agents.fundamentals import fundamentals_agent
from src.agents.growth import growth_agent
from src.agents.sentiment import sentiment_agent
from src.agents.technical import technical_agent
from src.agents.valuation import valuation_agent

# Registry: key -> (node_name, agent_function)
ANALYST_CONFIG: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "fundamentals": ("fundamentals_analyst", fundamentals_agent),
    "technical": ("technical_analyst", technical_agent),
    "sentiment": ("sentiment_analyst", sentiment_agent),
    "valuation": ("valuation_analyst", valuation_agent),
    "growth": ("growth_analyst", growth_agent),
}

# To add a new analyst:
# 1. Create src/agents/my_analyst.py following the same pattern
# 2. Add entry here: "my_key": ("my_node_name", my_agent_function)
