"""Warren Buffett persona agent — value investing with margin of safety."""
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

AGENT_ID = "buffett_analyst"

SYSTEM_PROMPT = (
    "You are Warren Buffett, the legendary value investor. Analyze stocks using your core principles:\n\n"
    "Investment Philosophy:\n"
    "- Circle of competence: Only invest in businesses you understand\n"
    "- Competitive moat: Look for durable advantages (brand, switching costs, scale)\n"
    "- Management quality: Honest, capable operators who allocate capital well\n"
    "- Financial strength: Low debt, high ROE, consistent earnings\n"
    "- Margin of safety: Buy wonderful businesses at fair prices, or fair businesses at wonderful prices\n"
    "- Long-term horizon: Think like a business owner, not a stock trader\n\n"
    "Signal Rules:\n"
    "- BULLISH: Strong business with competitive moat AND trading below intrinsic value (margin of safety > 0)\n"
    "- BEARISH: Weak fundamentals, deteriorating moat, OR significantly overvalued\n"
    "- NEUTRAL: Good business but no margin of safety, or insufficient data to judge\n\n"
    "Confidence Scale:\n"
    "- 80-100: Exceptional business within circle of competence, clear margin of safety\n"
    "- 60-79: Good business with decent moat, reasonable valuation\n"
    "- 40-59: Mixed signals, would want better price or more information\n"
    "- 20-39: Outside expertise or concerning fundamentals\n"
    "- 0-19: Poor business or significantly overvalued\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Buffett would."
)


def buffett_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Warren Buffett's value investing lens.

    Always uses LLM — persona agents require an API key.
    """
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
    """Analyze a ticker through Buffett's lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=8)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for Buffett-style analysis.",
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
            reasoning="Unable to complete Buffett-style analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string from financial data."""
    latest = metrics[0]
    facts: list[str] = []

    # ROE consistency
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roes:
        high_roe_count = sum(1 for r in roes if r > 0.15)
        facts.append(f"ROE: {roes[0]:.1%} (latest), {high_roe_count}/{len(roes)} periods above 15%")

    # Debt to equity
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f}")

    # Profit margins
    margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if margins:
        facts.append(f"Net Margin: {margins[0]:.1%} (latest)")
        if len(margins) >= 2:
            trend = "expanding" if margins[0] > margins[-1] else "contracting"
            facts.append(f"Margin Trend: {margins[-1]:.1%} → {margins[0]:.1%} ({trend})")

    # Earnings consistency
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        growing_periods = sum(1 for i in range(len(earnings) - 1) if earnings[i] > earnings[i + 1])
        facts.append(f"Earnings Growth: {growing_periods}/{len(earnings) - 1} periods growing")
        if earnings[-1] and earnings[-1] > 0:
            total_growth = (earnings[0] - earnings[-1]) / abs(earnings[-1])
            facts.append(f"Total Earnings Growth: {total_growth:.1%} over {len(earnings)} periods")

    # Book value per share
    if details:
        shares = details.weighted_shares_outstanding or details.share_class_shares_outstanding
        if shares and shares > 0:
            bvps = [
                m.shareholders_equity / shares
                for m in metrics
                if m.shareholders_equity is not None
            ]
            if len(bvps) >= 2:
                bv_growth = (bvps[0] - bvps[-1]) / abs(bvps[-1]) if bvps[-1] != 0 else 0
                facts.append(f"Book Value/Share: ${bvps[0]:,.2f} (growth: {bv_growth:.1%})")

    # Free cash flow
    if latest.operating_cash_flow is not None:
        facts.append(f"Operating Cash Flow: ${latest.operating_cash_flow / 1e9:.1f}B")
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")

    # Market cap and intrinsic value estimate
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")

        # Simple intrinsic value: 10x FCF or 15x earnings
        if latest.free_cash_flow and latest.free_cash_flow > 0:
            iv_fcf = latest.free_cash_flow * 10
            margin_of_safety = (iv_fcf - market_cap) / market_cap
            facts.append(f"Intrinsic Value (10x FCF): ${iv_fcf / 1e9:.1f}B "
                         f"(margin of safety: {margin_of_safety:.1%})")
        elif latest.net_income and latest.net_income > 0:
            iv_earn = latest.net_income * 15
            margin_of_safety = (iv_earn - market_cap) / market_cap
            facts.append(f"Intrinsic Value (15x earnings): ${iv_earn / 1e9:.1f}B "
                         f"(margin of safety: {margin_of_safety:.1%})")

    # Revenue growth
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B (growth: {rev_growth:.1%} over {len(revenues)} periods)")

    # Current ratio
    if latest.current_ratio is not None:
        facts.append(f"Current Ratio: {latest.current_ratio:.1f}")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
