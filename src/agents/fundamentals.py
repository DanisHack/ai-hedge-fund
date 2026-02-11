"""Fundamentals analyst agent."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, SignalType
from src.data.polygon_client import get_company_details, get_financial_metrics
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

AGENT_ID = "fundamentals_analyst"


def fundamentals_agent(state: AgentState) -> dict[str, Any]:
    """Analyze each ticker's fundamental financial data and generate signals.

    Scores: net margin, ROE, debt/equity, revenue growth.
    """
    data = state["data"]
    tickers: list[str] = data.get("tickers", [])
    end_date: str = data.get("end_date", "")
    show_reasoning: bool = state["metadata"].get("show_reasoning", False)

    signals: list[dict] = []

    for ticker in tickers:
        try:
            signal = _analyze_ticker(ticker, end_date)
        except Exception as e:
            logger.warning(f"[{AGENT_ID}] Failed to analyze {ticker}: {e}")
            signal = AnalystSignal(
                agent_id=AGENT_ID, ticker=ticker,
                signal=SignalType.NEUTRAL, confidence=0,
                reasoning=f"Analysis failed: {e}",
            )

        signals.append(signal.model_dump(mode="json"))

        if show_reasoning:
            logger.info(f"[{AGENT_ID}] {ticker}: {signal.signal.value} "
                        f"(confidence={signal.confidence}) — {signal.reasoning}")

    message = HumanMessage(
        content=json.dumps({"agent": AGENT_ID, "signals": signals}),
        name=AGENT_ID,
    )
    return {
        "messages": [message],
        "data": {"analyst_signals": {AGENT_ID: signals}},
    }


def _analyze_ticker(ticker: str, end_date: str) -> AnalystSignal:
    """Run fundamentals analysis on a single ticker."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=4)
    get_company_details(ticker)  # warm the cache for downstream use

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="No financial data available.",
        )

    latest = metrics[0]
    score = 0
    max_score = 0
    reasons: list[str] = []

    # --- Profitability ---
    if latest.net_profit_margin is not None:
        max_score += 2
        if latest.net_profit_margin > 0.15:
            score += 2
            reasons.append(f"Strong net margin: {latest.net_profit_margin:.1%}")
        elif latest.net_profit_margin > 0.05:
            score += 1
            reasons.append(f"Moderate net margin: {latest.net_profit_margin:.1%}")
        else:
            reasons.append(f"Weak net margin: {latest.net_profit_margin:.1%}")

    # --- Return on Equity ---
    if latest.return_on_equity is not None:
        max_score += 2
        if latest.return_on_equity > 0.15:
            score += 2
            reasons.append(f"Strong ROE: {latest.return_on_equity:.1%}")
        elif latest.return_on_equity > 0.08:
            score += 1
            reasons.append(f"Moderate ROE: {latest.return_on_equity:.1%}")
        else:
            reasons.append(f"Weak ROE: {latest.return_on_equity:.1%}")

    # --- Leverage ---
    if latest.debt_to_equity is not None:
        max_score += 2
        if latest.debt_to_equity < 0.5:
            score += 2
            reasons.append(f"Low leverage: D/E={latest.debt_to_equity:.2f}")
        elif latest.debt_to_equity < 1.5:
            score += 1
            reasons.append(f"Moderate leverage: D/E={latest.debt_to_equity:.2f}")
        else:
            reasons.append(f"High leverage: D/E={latest.debt_to_equity:.2f}")

    # --- Revenue Growth (compare last 2 periods) ---
    if len(metrics) >= 2 and metrics[0].revenue and metrics[1].revenue and metrics[1].revenue > 0:
        max_score += 2
        growth = (metrics[0].revenue - metrics[1].revenue) / metrics[1].revenue
        if growth > 0.10:
            score += 2
            reasons.append(f"Strong revenue growth: {growth:.1%}")
        elif growth > 0:
            score += 1
            reasons.append(f"Positive revenue growth: {growth:.1%}")
        else:
            reasons.append(f"Revenue decline: {growth:.1%}")

    # --- Determine signal ---
    if max_score == 0:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="Insufficient data for analysis.",
        )

    ratio = score / max_score
    confidence = round(ratio * 100)

    if ratio >= 0.65:
        signal = SignalType.BULLISH
    elif ratio <= 0.35:
        signal = SignalType.BEARISH
    else:
        signal = SignalType.NEUTRAL

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=signal, confidence=confidence,
        reasoning="; ".join(reasons),
    )
