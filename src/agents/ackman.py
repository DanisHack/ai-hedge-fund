"""Bill Ackman persona agent — activist investor who takes bold positions."""
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

AGENT_ID = "ackman_analyst"

SYSTEM_PROMPT = (
    "You are Bill Ackman, the activist hedge fund manager of Pershing Square. "
    "Analyze stocks using your activist investing framework:\n\n"
    "Investment Philosophy:\n"
    "- Invest in high-quality businesses with durable competitive advantages\n"
    "- Look for underperforming companies where activist engagement can unlock value\n"
    "- Focus on free cash flow generation and capital allocation efficiency\n"
    "- Take concentrated, high-conviction positions — typically 8-12 holdings\n"
    "- Identify catalysts: operational improvements, cost cuts, strategic changes, or board changes\n"
    "- Prefer simple, predictable businesses with pricing power\n"
    "- Margin improvement potential is a key value driver\n\n"
    "Signal Rules:\n"
    "- BULLISH: Undervalued business with clear path to margin improvement or operational turnaround\n"
    "- BEARISH: Deteriorating fundamentals with no clear catalyst, or overvalued relative to peers\n"
    "- NEUTRAL: Fairly valued with limited upside from operational improvement\n\n"
    "Confidence Scale:\n"
    "- 80-100: High-quality business trading at a significant discount with clear catalyst\n"
    "- 60-79: Good business with some operational improvement opportunity\n"
    "- 40-59: Decent business but limited activist angle or uncertain catalyst\n"
    "- 20-39: Unclear fundamentals or limited margin improvement potential\n"
    "- 0-19: Poor business quality or significantly overvalued\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Ackman would, with a bold and conviction-driven tone."
)


def ackman_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Ackman's activist investing lens."""
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
    """Analyze a ticker through Ackman's activist lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for activist analysis.",
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
            reasoning="Unable to complete Ackman-style activist analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on activist value creation."""
    latest = metrics[0]
    facts: list[str] = []

    # Margin analysis — key for activist thesis
    gross_margins = [m.gross_profit_margin for m in metrics if m.gross_profit_margin is not None]
    if gross_margins:
        facts.append(f"Gross Margin: {gross_margins[0]:.1%} (latest)")
        if len(gross_margins) >= 2:
            trend = "expanding" if gross_margins[0] > gross_margins[-1] else "contracting"
            facts.append(f"Gross Margin Trend: {gross_margins[-1]:.1%} → {gross_margins[0]:.1%} ({trend})")

    net_margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if net_margins:
        facts.append(f"Net Margin: {net_margins[0]:.1%} (latest)")
        if len(net_margins) >= 2:
            margin_gap = net_margins[0] - net_margins[-1]
            facts.append(f"Net Margin Change: {margin_gap:+.1%} over {len(net_margins)} periods")

    # Margin gap (gross - net) = operational inefficiency opportunity
    if gross_margins and net_margins:
        gap = gross_margins[0] - net_margins[0]
        facts.append(f"Gross-to-Net Margin Gap: {gap:.1%} (operational improvement opportunity)")

    # Free cash flow — Ackman loves FCF generators
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")
    if latest.operating_cash_flow is not None:
        facts.append(f"Operating Cash Flow: ${latest.operating_cash_flow / 1e9:.1f}B")

    # FCF yield
    market_cap = details.market_cap if details else None
    if market_cap and latest.free_cash_flow and latest.free_cash_flow > 0:
        fcf_yield = latest.free_cash_flow / market_cap
        facts.append(f"FCF Yield: {fcf_yield:.1%}")

    # Revenue and growth
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B (latest)")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            facts.append(f"Revenue Growth: {rev_growth:.1%} over {len(revenues)} periods")

    # Capital structure
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f}")
    if latest.total_liabilities and latest.total_assets and latest.total_assets > 0:
        leverage = latest.total_liabilities / latest.total_assets
        facts.append(f"Leverage (Liabilities/Assets): {leverage:.1%}")

    # ROE — capital allocation efficiency
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roes:
        facts.append(f"ROE: {roes[0]:.1%} (latest)")
        if len(roes) >= 2:
            roe_trend = "improving" if roes[0] > roes[-1] else "declining"
            facts.append(f"ROE Trend: {roes[-1]:.1%} → {roes[0]:.1%} ({roe_trend})")

    # Market cap
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")

    # EPS trend
    eps_vals = [m.earnings_per_share for m in metrics if m.earnings_per_share is not None]
    if len(eps_vals) >= 2:
        eps_growth = (eps_vals[0] - eps_vals[-1]) / abs(eps_vals[-1]) if eps_vals[-1] != 0 else 0
        facts.append(f"EPS: ${eps_vals[0]:.2f} (growth: {eps_growth:.1%})")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
