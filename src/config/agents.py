"""Agent registry for LangGraph workflow configuration."""
from __future__ import annotations

from typing import Any, Callable

from src.agents.buffett import buffett_agent
from src.agents.fundamentals import fundamentals_agent
from src.agents.graham import graham_agent
from src.agents.growth import growth_agent
from src.agents.macro_regime import macro_regime_agent
from src.agents.sentiment import sentiment_agent
from src.agents.technical import technical_agent
from src.agents.valuation import valuation_agent

# Core analysts: always run (rule-based by default, LLM-powered with --use-llm)
ANALYST_CONFIG: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "fundamentals": ("fundamentals_analyst", fundamentals_agent),
    "technical": ("technical_analyst", technical_agent),
    "sentiment": ("sentiment_analyst", sentiment_agent),
    "valuation": ("valuation_analyst", valuation_agent),
    "growth": ("growth_analyst", growth_agent),
    "macro_regime": ("macro_regime_analyst", macro_regime_agent),
}

# Persona agents: opt-in via --personas flag (LLM-only, require API key)
PERSONA_CONFIG: dict[str, tuple[str, Callable[..., dict[str, Any]]]] = {
    "buffett": ("buffett_analyst", buffett_agent),
    "graham": ("graham_analyst", graham_agent),
}
