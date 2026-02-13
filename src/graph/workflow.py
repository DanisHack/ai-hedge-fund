"""LangGraph workflow builder and runner."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agents.portfolio_manager import portfolio_manager_agent
from src.agents.risk_manager import risk_manager_agent
from src.config.agents import ANALYST_CONFIG, PERSONA_CONFIG
from src.data.polygon_client import prefetch_ticker_data
from src.graph.state import AgentState

logger = logging.getLogger(__name__)


def create_workflow(personas: list[str] | None = None) -> StateGraph:
    """Build the LangGraph workflow.

    Topology:
        START ──┬── analyst_1 ──┐
                ├── analyst_2 ──┤
                ├── persona_1 ──┤
                └── ...       ──┤
                                ├── risk_manager ── portfolio_manager ── END
    """
    workflow = StateGraph(AgentState)

    # Register core analyst nodes
    analyst_node_names: list[str] = []
    for key, (node_name, agent_func) in ANALYST_CONFIG.items():
        workflow.add_node(node_name, agent_func)
        analyst_node_names.append(node_name)
        logger.debug(f"Registered analyst node: {node_name}")

    # Register persona nodes (opt-in)
    if personas:
        available = set(PERSONA_CONFIG.keys())
        requested = set(personas) if personas != ["all"] else available
        for key in requested & available:
            node_name, agent_func = PERSONA_CONFIG[key]
            workflow.add_node(node_name, agent_func)
            analyst_node_names.append(node_name)
            logger.debug(f"Registered persona node: {node_name}")
        unknown = requested - available
        if unknown:
            logger.warning(f"Unknown personas ignored: {unknown}. Available: {available}")

    # Register risk manager and portfolio manager
    workflow.add_node("risk_manager", risk_manager_agent)
    workflow.add_node("portfolio_manager", portfolio_manager_agent)

    # Edges: START -> all analysts (parallel fan-out)
    for node_name in analyst_node_names:
        workflow.add_edge(START, node_name)

    # Edges: all analysts -> risk_manager (fan-in)
    for node_name in analyst_node_names:
        workflow.add_edge(node_name, "risk_manager")

    # Edges: risk_manager -> portfolio_manager -> END
    workflow.add_edge("risk_manager", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    return workflow


def run_hedge_fund(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    portfolio: dict[str, Any] | None = None,
    model_name: str = "gpt-4o-mini",
    model_provider: str = "openai",
    show_reasoning: bool = True,
    use_llm: bool = False,
    personas: list[str] | None = None,
) -> dict[str, Any]:
    """Execute the full hedge fund workflow.

    Returns the final AgentState with all signals and portfolio decisions.
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    if portfolio is None:
        portfolio = {"cash": 100000, "positions": {}, "total_value": 100000}

    workflow = create_workflow(personas=personas)
    graph = workflow.compile()

    initial_state: AgentState = {
        "messages": [],
        "data": {
            "tickers": tickers,
            "portfolio": portfolio,
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
            "current_prices": {},
        },
        "metadata": {
            "model_name": model_name,
            "model_provider": model_provider,
            "show_reasoning": show_reasoning,
            "use_llm": use_llm,
        },
    }

    logger.info(f"Running hedge fund for {tickers} ({start_date} to {end_date})")

    # Pre-fetch all data so agents hit cache instead of flooding the API
    logger.info(f"Prefetching market data for {tickers}...")
    prefetch_ticker_data(tickers, start_date, end_date)

    final_state = graph.invoke(initial_state)
    logger.info("Workflow complete.")

    return final_state
