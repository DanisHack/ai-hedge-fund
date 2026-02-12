"""Rakesh Jhunjhunwala persona agent — India's Big Bull growth investor."""
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

AGENT_ID = "jhunjhunwala_analyst"

SYSTEM_PROMPT = (
    "You are Rakesh Jhunjhunwala, India's legendary 'Big Bull' investor. "
    "Analyze stocks using your growth-oriented, conviction-driven framework:\n\n"
    "Investment Philosophy:\n"
    "- Be bullish on growth: invest in countries and companies that are growing\n"
    "- Buy market leaders with dominant positions in expanding industries\n"
    "- Earnings acceleration is the strongest buy signal — rising earnings attract rising prices\n"
    "- Strong ROE indicates a well-managed business with competitive advantages\n"
    "- Reasonable leverage is acceptable if the business can service debt from cash flows\n"
    "- Take large, concentrated bets on high-conviction ideas\n"
    "- Hold winners through volatility — let compounding work\n"
    "- Combination of top-down macro view with bottom-up stock selection\n"
    "- Avoid companies with declining market share or margin compression\n\n"
    "Signal Rules:\n"
    "- BULLISH: Accelerating earnings, market leadership, strong ROE, and reasonable valuation\n"
    "- BEARISH: Decelerating earnings, margin compression, or overvalued relative to growth\n"
    "- NEUTRAL: Moderate growth with unclear acceleration or mixed fundamentals\n\n"
    "Confidence Scale:\n"
    "- 80-100: Market leader with accelerating earnings and strong macro tailwinds\n"
    "- 60-79: Solid growth company with good management and expanding margins\n"
    "- 40-59: Average growth or uncertain market position\n"
    "- 20-39: Weak growth or margin pressure\n"
    "- 0-19: Declining business or excessive valuation\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Jhunjhunwala would, with a bold and optimistic tone."
)


def jhunjhunwala_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Jhunjhunwala's growth and conviction lens."""
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
    """Analyze a ticker through Jhunjhunwala's growth lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for growth analysis.",
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
            reasoning="Unable to complete Jhunjhunwala-style growth analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on growth acceleration and market leadership."""
    latest = metrics[0]
    facts: list[str] = []

    # Earnings acceleration — the primary Jhunjhunwala signal
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        if earnings[-1] and earnings[-1] > 0:
            earn_growth = (earnings[0] - earnings[-1]) / abs(earnings[-1])
            facts.append(f"Earnings Growth: {earn_growth:.1%} over {len(earnings)} periods")

        # Acceleration check
        if len(earnings) >= 3:
            recent = (earnings[0] - earnings[1]) / abs(earnings[1]) if earnings[1] and earnings[1] != 0 else 0
            prior = (earnings[1] - earnings[2]) / abs(earnings[2]) if earnings[2] and earnings[2] != 0 else 0
            if recent > prior:
                facts.append(f"Earnings ACCELERATING: {prior:.1%} → {recent:.1%}")
            else:
                facts.append(f"Earnings decelerating: {prior:.1%} → {recent:.1%}")

    # EPS
    eps_vals = [m.earnings_per_share for m in metrics if m.earnings_per_share is not None]
    if eps_vals:
        facts.append(f"EPS: ${eps_vals[0]:.2f} (latest)")

    # Revenue growth — market leadership indicator
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            facts.append(f"Revenue Growth: {rev_growth:.1%} over {len(revenues)} periods")

    # ROE — management quality
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roes:
        facts.append(f"ROE: {roes[0]:.1%} ({'strong' if roes[0] > 0.18 else 'moderate' if roes[0] > 0.12 else 'weak'})")
        if len(roes) >= 2:
            roe_trend = "improving" if roes[0] > roes[-1] else "declining"
            facts.append(f"ROE Trend: {roes[-1]:.1%} → {roes[0]:.1%} ({roe_trend})")

    # Margins
    net_margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if net_margins:
        facts.append(f"Net Margin: {net_margins[0]:.1%}")
        if len(net_margins) >= 2:
            trend = "expanding" if net_margins[0] > net_margins[-1] else "compressing"
            facts.append(f"Margin Trend: {net_margins[-1]:.1%} → {net_margins[0]:.1%} ({trend})")

    # Debt management
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f}")

    # Cash flow
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")
    if latest.operating_cash_flow is not None:
        facts.append(f"Operating Cash Flow: ${latest.operating_cash_flow / 1e9:.1f}B")

    # Market cap and valuation
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")
        if latest.net_income and latest.net_income > 0:
            pe = market_cap / latest.net_income
            facts.append(f"P/E Ratio: {pe:.1f}")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
