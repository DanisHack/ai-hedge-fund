"""Mohnish Pabrai persona agent — the Dhandho investor seeking low-risk doubles."""
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

AGENT_ID = "pabrai_analyst"

SYSTEM_PROMPT = (
    "You are Mohnish Pabrai, the Dhandho investor and author of 'The Dhandho Investor'. "
    "Analyze stocks using your low-risk, high-return framework:\n\n"
    "Investment Philosophy:\n"
    "- Dhandho: 'Heads I win big, tails I don't lose much' — seek asymmetric risk/reward\n"
    "- Clone successful investors — study what the best minds are buying\n"
    "- Focus on simple, predictable businesses with durable moats\n"
    "- Buy when there's maximum pessimism — temporary fear creates opportunity\n"
    "- Concentrated portfolio: few bets, big bets, infrequent bets\n"
    "- Low downside risk is more important than high upside potential\n"
    "- Look for businesses trading at a fraction of intrinsic value\n"
    "- Patience: wait for the perfect pitch then swing hard\n\n"
    "Signal Rules:\n"
    "- BULLISH: Low downside (strong balance sheet, cheap valuation) with significant upside potential\n"
    "- BEARISH: Weak balance sheet, high valuation, or unfavorable risk/reward asymmetry\n"
    "- NEUTRAL: Decent business but risk/reward is not asymmetrically favorable\n\n"
    "Confidence Scale:\n"
    "- 80-100: Extreme asymmetry — minimal downside with 2-3x upside potential\n"
    "- 60-79: Good risk/reward with solid margin of safety\n"
    "- 40-59: Some value but asymmetry is unclear\n"
    "- 20-39: Risk/reward not compelling enough for a concentrated bet\n"
    "- 0-19: Poor risk/reward — too much downside exposure\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Pabrai would, with a calm and patient tone."
)


def pabrai_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Pabrai's Dhandho investing lens."""
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
    """Analyze a ticker through Pabrai's Dhandho lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for Dhandho analysis.",
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
            reasoning="Unable to complete Pabrai-style Dhandho analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on asymmetric risk/reward."""
    latest = metrics[0]
    facts: list[str] = []

    # Downside protection — balance sheet strength
    if latest.current_ratio is not None:
        facts.append(f"Current Ratio: {latest.current_ratio:.2f} "
                     f"({'strong floor' if latest.current_ratio >= 2.0 else 'adequate' if latest.current_ratio >= 1.5 else 'risky'})")
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f} "
                     f"({'low risk' if latest.debt_to_equity < 0.3 else 'moderate' if latest.debt_to_equity < 1.0 else 'high risk'})")

    # Asset backing
    if latest.total_assets and latest.total_liabilities:
        net_assets = latest.total_assets - latest.total_liabilities
        facts.append(f"Net Asset Value: ${net_assets / 1e9:.1f}B")

    # Valuation — cheapness matters
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")

        if latest.net_income and latest.net_income > 0:
            pe = market_cap / latest.net_income
            facts.append(f"P/E Ratio: {pe:.1f} ({'cheap' if pe < 10 else 'reasonable' if pe < 18 else 'expensive'})")

        if latest.free_cash_flow and latest.free_cash_flow > 0:
            fcf_yield = latest.free_cash_flow / market_cap
            facts.append(f"FCF Yield: {fcf_yield:.1%} ({'excellent' if fcf_yield > 0.08 else 'good' if fcf_yield > 0.05 else 'fair'})")

            # Upside estimate: if FCF yield normalizes to 5%, what's the implied market cap?
            implied_value = latest.free_cash_flow / 0.05
            upside = (implied_value - market_cap) / market_cap
            if upside > 0:
                facts.append(f"Upside to Fair Value (5% FCF yield): {upside:.0%}")

        if latest.shareholders_equity and latest.shareholders_equity > 0:
            pb = market_cap / latest.shareholders_equity
            facts.append(f"Price/Book: {pb:.2f}")

    # Earnings consistency — predictability matters
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        positive = sum(1 for e in earnings if e and e > 0)
        growing = sum(1 for i in range(len(earnings) - 1) if earnings[i] > earnings[i + 1])
        facts.append(f"Earnings: positive {positive}/{len(earnings)} periods, growing {growing}/{len(earnings) - 1}")

    # ROE — quality of earnings
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roes:
        facts.append(f"ROE: {roes[0]:.1%} (latest)")

    # Revenue
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            facts.append(f"Revenue Growth: {rev_growth:.1%}")

    # Free cash flow
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
