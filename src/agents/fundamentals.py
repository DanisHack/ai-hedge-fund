"""Fundamentals analyst agent."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, LLMAnalysisResult, SignalType
from src.data.polygon_client import get_company_details, get_financial_metrics
from src.graph.state import AgentState
from src.llm import call_llm

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
            signal = _analyze_ticker(ticker, end_date, metadata=state["metadata"])
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


def _analyze_ticker(ticker: str, end_date: str, metadata: dict | None = None) -> AnalystSignal:
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
    analysis_data: dict[str, Any] = {}

    # --- Profitability ---
    if latest.net_profit_margin is not None:
        max_score += 2
        analysis_data["net_margin"] = latest.net_profit_margin
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
        analysis_data["return_on_equity"] = latest.return_on_equity
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
        analysis_data["debt_to_equity"] = latest.debt_to_equity
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
        analysis_data["revenue_growth"] = growth
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

    rule_based = AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=signal, confidence=confidence,
        reasoning="; ".join(reasons),
    )

    if metadata and metadata.get("use_llm") and analysis_data:
        return _llm_analyze(ticker, analysis_data, rule_based, metadata)

    return rule_based


def _llm_analyze(
    ticker: str,
    analysis_data: dict[str, Any],
    rule_based: AnalystSignal,
    metadata: dict,
) -> AnalystSignal:
    """Use LLM to reason about fundamentals data."""
    facts = []
    if "net_margin" in analysis_data:
        facts.append(f"- Net Profit Margin: {analysis_data['net_margin']:.1%}")
    if "return_on_equity" in analysis_data:
        facts.append(f"- Return on Equity: {analysis_data['return_on_equity']:.1%}")
    if "debt_to_equity" in analysis_data:
        facts.append(f"- Debt to Equity: {analysis_data['debt_to_equity']:.2f}")
    if "revenue_growth" in analysis_data:
        facts.append(f"- Revenue Growth (QoQ): {analysis_data['revenue_growth']:.1%}")

    prompt = (
        f"You are a fundamentals analyst evaluating {ticker}.\n\n"
        f"Financial Metrics:\n"
        + "\n".join(facts)
        + f"\n\nRule-based score: {rule_based.confidence}% ({rule_based.signal.value})\n\n"
        "Analyze these fundamentals. Consider:\n"
        "1. Profitability: Is the margin healthy and sustainable?\n"
        "2. Capital efficiency: Is ROE competitive?\n"
        "3. Financial risk: Is leverage manageable?\n"
        "4. Revenue momentum: Is the business growing?\n"
        "5. Metric interactions (e.g., high margins with rising debt may be unsustainable)\n\n"
        "Provide a trading signal (bullish/bearish/neutral), confidence 0-100, "
        "and 2-4 sentence reasoning citing specific data points."
    )

    result = call_llm(
        prompt=prompt,
        response_model=LLMAnalysisResult,
        model_name=metadata.get("model_name", "gpt-4o-mini"),
        model_provider=metadata.get("model_provider", "openai"),
        default_factory=lambda: LLMAnalysisResult(
            signal=rule_based.signal,
            confidence=rule_based.confidence,
            reasoning=rule_based.reasoning,
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )
