"""Cathie Wood persona agent — growth investing focused on innovation and disruption."""
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

AGENT_ID = "wood_analyst"

SYSTEM_PROMPT = (
    "You are Cathie Wood, founder and CEO of ARK Invest. "
    "Analyze stocks using your innovation-driven investment framework:\n\n"
    "Investment Philosophy:\n"
    "- Invest in disruptive innovation across technology platforms\n"
    "- Focus on exponential growth potential — 5-year time horizon minimum\n"
    "- Key themes: AI, robotics, genomics, fintech, energy storage, blockchain\n"
    "- Revenue growth rate is the most important metric — prioritize top-line acceleration\n"
    "- Gross margins indicate scalability and pricing power of the platform\n"
    "- Willingness to accept near-term losses for transformational long-term upside\n"
    "- Total addressable market (TAM) expansion is critical — look for companies creating new markets\n"
    "- Conviction in innovation: short-term volatility creates buying opportunities\n\n"
    "Signal Rules:\n"
    "- BULLISH: High revenue growth, expanding TAM, strong gross margins, positioned in disruptive theme\n"
    "- BEARISH: Decelerating growth, shrinking margins, legacy business model with no innovation pivot\n"
    "- NEUTRAL: Moderate growth but unclear disruption thesis, or fairly valued relative to growth\n\n"
    "Confidence Scale:\n"
    "- 80-100: Transformational company with accelerating revenue and massive TAM\n"
    "- 60-79: Strong growth with clear innovation angle\n"
    "- 40-59: Some growth potential but limited disruption characteristics\n"
    "- 20-39: Mature business with slowing growth\n"
    "- 0-19: Legacy business with no innovation thesis\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Cathie Wood would, with an optimistic and visionary tone."
)


def wood_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Cathie Wood's innovation and disruption lens."""
    data = state["data"]
    tickers: list[str] = data.get("tickers", [])
    end_date: str = data.get("end_date", "")
    metadata: dict = state["metadata"]
    show_reasoning: bool = metadata.get("show_reasoning", False)

    signals: list[dict] = []

    for ticker in tickers:
        try:
            signal = _analyze_ticker(ticker, end_date, metadata)
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


def _analyze_ticker(ticker: str, end_date: str, metadata: dict) -> AnalystSignal:
    """Analyze a ticker through Wood's innovation lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for innovation analysis.",
        )

    facts = _build_facts(metrics, details)

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Analyze {ticker} based on these financial facts:\n\n{facts}"
    )

    result = call_llm(
        prompt=prompt,
        response_model=LLMAnalysisResult,
        model_name=metadata.get("model_name", "gpt-4o-mini"),
        model_provider=metadata.get("model_provider", "openai"),
        default_factory=lambda: LLMAnalysisResult(
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="Unable to complete innovation-focused analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on growth and innovation metrics."""
    latest = metrics[0]
    facts: list[str] = []

    # Revenue growth — the primary metric
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B (latest)")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            facts.append(f"Revenue Growth: {rev_growth:.1%} over {len(revenues)} periods")

        # Growth acceleration/deceleration
        if len(revenues) >= 3:
            recent_growth = (revenues[0] - revenues[1]) / abs(revenues[1]) if revenues[1] else 0
            prior_growth = (revenues[1] - revenues[2]) / abs(revenues[2]) if revenues[2] else 0
            if recent_growth > prior_growth:
                facts.append(f"Revenue Growth Acceleration: {prior_growth:.1%} → {recent_growth:.1%} (ACCELERATING)")
            else:
                facts.append(f"Revenue Growth Deceleration: {prior_growth:.1%} → {recent_growth:.1%} (decelerating)")

    # Gross margins — scalability indicator
    gross_margins = [m.gross_profit_margin for m in metrics if m.gross_profit_margin is not None]
    if gross_margins:
        facts.append(f"Gross Margin: {gross_margins[0]:.1%} (latest)")
        if len(gross_margins) >= 2:
            trend = "expanding" if gross_margins[0] > gross_margins[-1] else "contracting"
            facts.append(f"Gross Margin Trend: {gross_margins[-1]:.1%} → {gross_margins[0]:.1%} ({trend})")

    # Net margin trajectory
    net_margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if net_margins:
        facts.append(f"Net Margin: {net_margins[0]:.1%} (latest)")
        if net_margins[0] < 0:
            facts.append("Company is pre-profit — focus on revenue growth trajectory and gross margin")

    # R&D proxy: operating cash flow vs free cash flow gap
    if latest.operating_cash_flow and latest.free_cash_flow and latest.revenue:
        capex = latest.operating_cash_flow - latest.free_cash_flow
        capex_intensity = capex / latest.revenue
        facts.append(f"Capital Intensity (capex/revenue): {capex_intensity:.1%} "
                     f"({'high reinvestment' if capex_intensity > 0.10 else 'moderate' if capex_intensity > 0.05 else 'low'})")

    # Earnings growth
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        positive = sum(1 for e in earnings if e and e > 0)
        facts.append(f"Profitable in {positive}/{len(earnings)} periods")

    # Market cap — for context on where in growth cycle
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")
        if revenues:
            ps_ratio = market_cap / revenues[0]
            facts.append(f"Price/Sales: {ps_ratio:.1f}x")

    # Employee count — growth signal
    if details and details.total_employees:
        facts.append(f"Employees: {details.total_employees:,}")

    # FCF
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
