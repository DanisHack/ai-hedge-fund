"""Aswath Damodaran persona agent — the Dean of Valuation."""
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

AGENT_ID = "damodaran_analyst"

SYSTEM_PROMPT = (
    "You are Aswath Damodaran, the 'Dean of Valuation' and professor at NYU Stern. "
    "Analyze stocks using your disciplined valuation framework:\n\n"
    "Investment Philosophy:\n"
    "- Every asset has an intrinsic value driven by cash flows, growth, and risk\n"
    "- Story must match the numbers: a narrative without data is a fairy tale, data without narrative is noise\n"
    "- Use DCF as the primary tool: estimate cash flows, growth rate, and discount rate\n"
    "- ROIC vs WACC: value creation only happens when returns exceed the cost of capital\n"
    "- Be honest about uncertainty — use ranges, not false precision\n"
    "- Price is what you pay, value is what you get — they often diverge\n"
    "- Beware of accounting games: focus on operating income and reinvestment\n\n"
    "Signal Rules:\n"
    "- BULLISH: Intrinsic value materially above market price, story is consistent with numbers\n"
    "- BEARISH: Overvalued relative to cash flows, growth priced in exceeds what's plausible\n"
    "- NEUTRAL: Fairly valued, or story and numbers don't align clearly\n\n"
    "Confidence Scale:\n"
    "- 80-100: Clear mispricing with strong cash flow support and coherent narrative\n"
    "- 60-79: Likely undervalued but some uncertainty in growth assumptions\n"
    "- 40-59: Fairly valued or mixed signals between story and numbers\n"
    "- 20-39: Limited data or unclear narrative\n"
    "- 0-19: No basis for valuation\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Damodaran would, with an academic yet practical tone."
)


def damodaran_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Damodaran's valuation framework."""
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
    """Analyze a ticker through Damodaran's valuation lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for valuation analysis.",
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
            reasoning="Unable to complete Damodaran-style valuation analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on valuation fundamentals."""
    latest = metrics[0]
    facts: list[str] = []

    # Revenue and growth trajectory
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B (latest)")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            rev_cagr = (revenues[0] / revenues[-1]) ** (1 / len(revenues)) - 1 if revenues[-1] > 0 else 0
            facts.append(f"Revenue Growth: {rev_growth:.1%} total over {len(revenues)} periods, CAGR ~{rev_cagr:.1%}")

    # Operating income and margins
    if latest.net_income is not None and latest.revenue is not None and latest.revenue > 0:
        operating_margin = latest.net_income / latest.revenue
        facts.append(f"Operating Margin (net): {operating_margin:.1%}")

    margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if len(margins) >= 2:
        trend = "expanding" if margins[0] > margins[-1] else "contracting"
        facts.append(f"Margin Trend: {margins[-1]:.1%} → {margins[0]:.1%} ({trend})")

    # Free cash flow — the key Damodaran metric
    fcfs = [m.free_cash_flow for m in metrics if m.free_cash_flow is not None]
    if fcfs:
        facts.append(f"Free Cash Flow: ${fcfs[0] / 1e9:.1f}B (latest)")
        if latest.revenue and latest.revenue > 0:
            fcf_margin = fcfs[0] / latest.revenue
            facts.append(f"FCF Margin: {fcf_margin:.1%}")
        if len(fcfs) >= 2 and fcfs[-1] and fcfs[-1] > 0:
            fcf_growth = (fcfs[0] - fcfs[-1]) / abs(fcfs[-1])
            facts.append(f"FCF Growth: {fcf_growth:.1%} over {len(fcfs)} periods")

    # ROIC proxy (ROE as available metric)
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roes:
        facts.append(f"ROE: {roes[0]:.1%} (latest)")
        if len(roes) >= 2:
            avg_roe = sum(roes) / len(roes)
            facts.append(f"Average ROE: {avg_roe:.1%} over {len(roes)} periods")

    # Reinvestment rate proxy
    if latest.operating_cash_flow and latest.free_cash_flow:
        capex = latest.operating_cash_flow - latest.free_cash_flow
        if latest.operating_cash_flow > 0:
            reinvestment_rate = capex / latest.operating_cash_flow
            facts.append(f"Reinvestment Rate (capex/OCF): {reinvestment_rate:.1%}")

    # Debt and cost of capital indicators
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f}")

    # Market cap and implied valuation
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")
        if fcfs and fcfs[0] and fcfs[0] > 0:
            fcf_yield = fcfs[0] / market_cap
            implied_multiple = market_cap / fcfs[0]
            facts.append(f"FCF Yield: {fcf_yield:.1%} (implied {implied_multiple:.1f}x FCF)")

            # Simple DCF: 10-year FCF at estimated growth, 10% discount
            growth_rate = 0.05  # conservative 5%
            terminal_growth = 0.03
            discount_rate = 0.10
            pv_fcf = sum(fcfs[0] * (1 + growth_rate) ** i / (1 + discount_rate) ** i for i in range(1, 11))
            terminal_value = fcfs[0] * (1 + growth_rate) ** 10 * (1 + terminal_growth) / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** 10
            intrinsic_value = pv_fcf + pv_terminal
            margin_of_safety = (intrinsic_value - market_cap) / market_cap
            facts.append(f"DCF Estimate (5% growth, 10% WACC): ${intrinsic_value / 1e9:.1f}B "
                         f"(margin of safety: {margin_of_safety:.1%})")

    # Earnings per share
    if latest.earnings_per_share is not None:
        facts.append(f"EPS: ${latest.earnings_per_share:.2f}")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
