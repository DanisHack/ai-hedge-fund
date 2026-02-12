"""Growth analyst agent — revenue/earnings trajectory and acceleration."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, LLMAnalysisResult, SignalType
from src.data.polygon_client import get_financial_metrics
from src.graph.state import AgentState
from src.llm import call_llm

logger = logging.getLogger(__name__)

AGENT_ID = "growth_analyst"


def growth_agent(state: AgentState) -> dict[str, Any]:
    """Analyze revenue and earnings growth trajectory for each ticker.

    Looks at: growth rate, consistency, acceleration/deceleration,
    and margin expansion.
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
    """Run growth analysis on a single ticker."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=8)

    if len(metrics) < 2:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="Insufficient historical data for growth analysis.",
        )

    score = 0
    max_score = 0
    reasons: list[str] = []
    analysis_data: dict[str, Any] = {}

    # --- 1. Revenue Growth Rate ---
    rev_growth_rates = _compute_growth_rates([m.revenue for m in metrics])
    if rev_growth_rates:
        max_score += 3
        latest_rev_growth = rev_growth_rates[0]
        avg_rev_growth = sum(rev_growth_rates) / len(rev_growth_rates)
        analysis_data["latest_rev_growth"] = latest_rev_growth
        analysis_data["avg_rev_growth"] = avg_rev_growth

        if latest_rev_growth > 0.15:
            score += 3
            reasons.append(f"Strong revenue growth: {latest_rev_growth:.1%} latest, {avg_rev_growth:.1%} avg")
        elif latest_rev_growth > 0.05:
            score += 2
            reasons.append(f"Moderate revenue growth: {latest_rev_growth:.1%} latest, {avg_rev_growth:.1%} avg")
        elif latest_rev_growth > 0:
            score += 1
            reasons.append(f"Slow revenue growth: {latest_rev_growth:.1%} latest")
        else:
            reasons.append(f"Revenue declining: {latest_rev_growth:.1%} latest")

    # --- 2. Earnings Growth Rate ---
    earnings_growth_rates = _compute_growth_rates([m.net_income for m in metrics])
    if earnings_growth_rates:
        max_score += 3
        latest_earn_growth = earnings_growth_rates[0]
        analysis_data["latest_earn_growth"] = latest_earn_growth

        if latest_earn_growth > 0.20:
            score += 3
            reasons.append(f"Strong earnings growth: {latest_earn_growth:.1%}")
        elif latest_earn_growth > 0.05:
            score += 2
            reasons.append(f"Moderate earnings growth: {latest_earn_growth:.1%}")
        elif latest_earn_growth > 0:
            score += 1
            reasons.append(f"Slow earnings growth: {latest_earn_growth:.1%}")
        else:
            reasons.append(f"Earnings declining: {latest_earn_growth:.1%}")

    # --- 3. Growth Acceleration/Deceleration ---
    if len(rev_growth_rates) >= 2:
        max_score += 2
        acceleration = rev_growth_rates[0] - rev_growth_rates[1]
        analysis_data["acceleration"] = acceleration

        if acceleration > 0.02:
            score += 2
            reasons.append(f"Revenue growth accelerating (+{acceleration:.1%}pp)")
        elif acceleration > -0.02:
            score += 1
            reasons.append(f"Revenue growth stable ({acceleration:+.1%}pp)")
        else:
            reasons.append(f"Revenue growth decelerating ({acceleration:+.1%}pp)")

    # --- 4. Growth Consistency ---
    if len(rev_growth_rates) >= 3:
        max_score += 2
        positive_periods = sum(1 for r in rev_growth_rates if r > 0)
        consistency = positive_periods / len(rev_growth_rates)
        analysis_data["consistency"] = consistency
        analysis_data["positive_periods"] = positive_periods
        analysis_data["total_periods"] = len(rev_growth_rates)

        if consistency >= 0.8:
            score += 2
            reasons.append(f"Consistent growth ({positive_periods}/{len(rev_growth_rates)} positive periods)")
        elif consistency >= 0.5:
            score += 1
            reasons.append(f"Mixed growth ({positive_periods}/{len(rev_growth_rates)} positive periods)")
        else:
            reasons.append(f"Inconsistent growth ({positive_periods}/{len(rev_growth_rates)} positive periods)")

    # --- 5. Margin Expansion ---
    margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if len(margins) >= 2:
        max_score += 2
        margin_change = margins[0] - margins[-1]
        analysis_data["margin_latest"] = margins[0]
        analysis_data["margin_oldest"] = margins[-1]
        analysis_data["margin_change"] = margin_change

        if margin_change > 0.02:
            score += 2
            reasons.append(f"Margin expanding: {margins[-1]:.1%} → {margins[0]:.1%}")
        elif margin_change > -0.02:
            score += 1
            reasons.append(f"Margin stable: {margins[0]:.1%}")
        else:
            reasons.append(f"Margin contracting: {margins[-1]:.1%} → {margins[0]:.1%}")

    # --- Determine signal ---
    if max_score == 0:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="Insufficient data for growth analysis.",
        )

    ratio = score / max_score
    confidence = round(ratio * 100)

    if ratio >= 0.65:
        signal = SignalType.BULLISH
    elif ratio <= 0.30:
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
    """Use LLM to reason about growth trajectory."""
    facts = []
    if "latest_rev_growth" in analysis_data:
        facts.append(f"- Revenue Growth (latest): {analysis_data['latest_rev_growth']:.1%}")
        facts.append(f"- Revenue Growth (avg): {analysis_data['avg_rev_growth']:.1%}")
    if "latest_earn_growth" in analysis_data:
        facts.append(f"- Earnings Growth (latest): {analysis_data['latest_earn_growth']:.1%}")
    if "acceleration" in analysis_data:
        facts.append(f"- Growth Acceleration: {analysis_data['acceleration']:+.1%}pp")
    if "consistency" in analysis_data:
        facts.append(f"- Consistency: {analysis_data['positive_periods']}/{analysis_data['total_periods']} positive periods")
    if "margin_change" in analysis_data:
        facts.append(f"- Margin Trend: {analysis_data['margin_oldest']:.1%} → {analysis_data['margin_latest']:.1%} "
                      f"(change: {analysis_data['margin_change']:+.1%})")

    prompt = (
        f"You are a growth analyst evaluating {ticker}.\n\n"
        f"Growth Metrics:\n"
        + "\n".join(facts)
        + f"\n\nRule-based score: {rule_based.confidence}% ({rule_based.signal.value})\n\n"
        "Analyze the growth trajectory. Consider:\n"
        "1. Growth quality: Is it driven by margin expansion or just top-line?\n"
        "2. Acceleration: Is growth speeding up or slowing down?\n"
        "3. Consistency: How reliable is the growth track record?\n"
        "4. Earnings vs revenue: Are earnings growing faster (operating leverage)?\n"
        "5. Sustainability: Can this growth rate be maintained?\n\n"
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


def _compute_growth_rates(values: list[float | None]) -> list[float]:
    """Compute period-over-period growth rates from a list of values.

    Values are ordered newest-first. Returns growth rates newest-first.
    """
    rates = []
    for i in range(len(values) - 1):
        current = values[i]
        previous = values[i + 1]
        if current is not None and previous is not None and previous != 0:
            rates.append((current - previous) / abs(previous))
    return rates
