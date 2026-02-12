"""Michael Burry persona agent — contrarian deep value investor."""
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

AGENT_ID = "burry_analyst"

SYSTEM_PROMPT = (
    "You are Michael Burry, the contrarian value investor famous for 'The Big Short'. "
    "Analyze stocks using your deep value and contrarian framework:\n\n"
    "Investment Philosophy:\n"
    "- Hunt for deep value: stocks trading well below intrinsic asset value\n"
    "- Be a contrarian — the best opportunities are in hated, ignored, or misunderstood companies\n"
    "- Focus on tangible book value, asset liquidation value, and enterprise value\n"
    "- Free cash flow yield is critical — look for high FCF relative to enterprise value\n"
    "- Balance sheet strength over income statement glamour\n"
    "- Short overvalued, speculative, or fraudulent companies when conviction is high\n"
    "- Be willing to go against the crowd — consensus is often wrong at extremes\n"
    "- Focus on what's measurable: cash, assets, debt, not narratives\n\n"
    "Signal Rules:\n"
    "- BULLISH: Trading significantly below tangible asset value or producing high FCF yield with strong balance sheet\n"
    "- BEARISH: Overvalued relative to tangible assets, cash burn, or speculative premium\n"
    "- NEUTRAL: Fairly valued or insufficient data to determine deep value opportunity\n\n"
    "Confidence Scale:\n"
    "- 80-100: Deep value with massive discount to tangible assets and strong FCF\n"
    "- 60-79: Clear undervaluation with decent balance sheet\n"
    "- 40-59: Some value characteristics but not deeply discounted\n"
    "- 20-39: Unclear value thesis or concerning balance sheet\n"
    "- 0-19: Overvalued or speculative — potential short candidate\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Burry would, with a data-driven and contrarian tone."
)


def burry_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Burry's contrarian deep value lens."""
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
    """Analyze a ticker through Burry's contrarian deep value lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for deep value analysis.",
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
            reasoning="Unable to complete Burry-style deep value analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on deep value and asset-based analysis."""
    latest = metrics[0]
    facts: list[str] = []

    shares = None
    if details:
        shares = details.weighted_shares_outstanding or details.share_class_shares_outstanding

    # Tangible book value
    if latest.total_assets is not None and latest.total_liabilities is not None:
        tangible_equity = latest.total_assets - latest.total_liabilities
        facts.append(f"Net Asset Value (Assets - Liabilities): ${tangible_equity / 1e9:.1f}B")
        facts.append(f"Total Assets: ${latest.total_assets / 1e9:.1f}B")
        facts.append(f"Total Liabilities: ${latest.total_liabilities / 1e9:.1f}B")

        if shares and shares > 0:
            tbvps = tangible_equity / shares
            facts.append(f"Tangible Book Value Per Share: ${tbvps:.2f}")

    # P/B ratio — key deep value metric
    market_cap = details.market_cap if details else None
    if market_cap and latest.shareholders_equity and latest.shareholders_equity > 0:
        pb = market_cap / latest.shareholders_equity
        facts.append(f"Price/Book: {pb:.2f} ({'deep value' if pb < 1.0 else 'below book' if pb < 1.5 else 'premium'})")

    # Enterprise value proxy
    if market_cap and latest.total_liabilities:
        cash_proxy = max(0, (latest.total_assets or 0) - (latest.total_liabilities or 0) - (latest.shareholders_equity or 0))
        ev = market_cap + latest.total_liabilities - cash_proxy
        facts.append(f"Enterprise Value (est): ${ev / 1e9:.1f}B")

    # FCF yield — Burry's key metric
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")
        if market_cap and latest.free_cash_flow > 0:
            fcf_yield = latest.free_cash_flow / market_cap
            facts.append(f"FCF Yield: {fcf_yield:.1%} "
                         f"({'very attractive' if fcf_yield > 0.10 else 'good' if fcf_yield > 0.06 else 'moderate'})")

    # P/E ratio
    if latest.earnings_per_share is not None:
        facts.append(f"EPS: ${latest.earnings_per_share:.2f}")
        if shares and market_cap and latest.earnings_per_share > 0:
            pe = (market_cap / shares) / latest.earnings_per_share
            facts.append(f"P/E Ratio: {pe:.1f} ({'deep value' if pe < 8 else 'value' if pe < 12 else 'growth premium'})")

    # Debt structure
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f}")
    if latest.current_ratio is not None:
        facts.append(f"Current Ratio: {latest.current_ratio:.2f} "
                     f"({'strong' if latest.current_ratio >= 2.0 else 'adequate' if latest.current_ratio >= 1.0 else 'WEAK'})")

    # Cash flow from operations
    if latest.operating_cash_flow is not None:
        facts.append(f"Operating Cash Flow: ${latest.operating_cash_flow / 1e9:.1f}B")

    # Market cap
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")

    # Revenue
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B")

    # Net margin
    if latest.net_profit_margin is not None:
        facts.append(f"Net Margin: {latest.net_profit_margin:.1%}")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
