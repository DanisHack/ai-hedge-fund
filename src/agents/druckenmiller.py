"""Stanley Druckenmiller persona agent — macro legend hunting asymmetric opportunities."""
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

AGENT_ID = "druckenmiller_analyst"

SYSTEM_PROMPT = (
    "You are Stanley Druckenmiller, the legendary macro investor who managed George Soros's "
    "Quantum Fund. Analyze stocks using your growth-at-scale, macro-aware framework:\n\n"
    "Investment Philosophy:\n"
    "- Look for asymmetric risk/reward: big upside with limited downside\n"
    "- Earnings momentum is the most important factor — buy companies with accelerating earnings\n"
    "- Macro context matters: align positions with prevailing economic trends\n"
    "- Concentrate on best ideas — when conviction is high, bet big\n"
    "- Focus on growth: the best returns come from companies growing into large markets\n"
    "- Liquidity and capital flows drive markets — follow the money\n"
    "- Be flexible: willing to change positions rapidly when thesis changes\n"
    "- Avoid permanent capital loss — size positions based on conviction and risk\n"
    "- The best trades combine strong fundamentals with favorable macro backdrop\n\n"
    "Signal Rules:\n"
    "- BULLISH: Accelerating earnings growth with macro tailwinds and reasonable valuation\n"
    "- BEARISH: Decelerating growth, unfavorable macro trends, or excessive valuation\n"
    "- NEUTRAL: Mixed growth signals or unclear macro positioning\n\n"
    "Confidence Scale:\n"
    "- 80-100: Accelerating growth with strong macro tailwinds — asymmetric opportunity\n"
    "- 60-79: Good growth fundamentals with supportive macro backdrop\n"
    "- 40-59: Moderate growth or mixed macro signals\n"
    "- 20-39: Decelerating growth or macro headwinds\n"
    "- 0-19: Poor fundamentals and unfavorable macro — potential short\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Druckenmiller would, with a decisive and macro-aware tone."
)


def druckenmiller_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Druckenmiller's macro-growth lens."""
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
    """Analyze a ticker through Druckenmiller's macro-growth lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for macro-growth analysis.",
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
            reasoning="Unable to complete Druckenmiller-style macro-growth analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on earnings momentum and macro context."""
    latest = metrics[0]
    facts: list[str] = []

    # Earnings momentum — Druckenmiller's primary signal
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        if earnings[-1] and earnings[-1] > 0:
            earn_growth = (earnings[0] - earnings[-1]) / abs(earnings[-1])
            facts.append(f"Earnings Growth: {earn_growth:.1%} over {len(earnings)} periods")

        # Acceleration — the key Druckenmiller signal
        if len(earnings) >= 3:
            recent = (earnings[0] - earnings[1]) / abs(earnings[1]) if earnings[1] and earnings[1] != 0 else 0
            prior = (earnings[1] - earnings[2]) / abs(earnings[2]) if earnings[2] and earnings[2] != 0 else 0
            if recent > prior and recent > 0:
                facts.append(f"Earnings ACCELERATING: {prior:.1%} → {recent:.1%} (strong buy signal)")
            elif recent < prior:
                facts.append(f"Earnings decelerating: {prior:.1%} → {recent:.1%} (caution)")

    # Revenue momentum
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            facts.append(f"Revenue Growth: {rev_growth:.1%} over {len(revenues)} periods")

        if len(revenues) >= 3:
            recent_rev = (revenues[0] - revenues[1]) / abs(revenues[1]) if revenues[1] else 0
            prior_rev = (revenues[1] - revenues[2]) / abs(revenues[2]) if revenues[2] else 0
            trend = "accelerating" if recent_rev > prior_rev else "decelerating"
            facts.append(f"Revenue Momentum: {trend}")

    # Margin trajectory
    net_margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if net_margins:
        facts.append(f"Net Margin: {net_margins[0]:.1%} (latest)")
        if len(net_margins) >= 2:
            expanding = net_margins[0] > net_margins[-1]
            facts.append(f"Margin Trend: {'expanding' if expanding else 'contracting'}")

    gross_margins = [m.gross_profit_margin for m in metrics if m.gross_profit_margin is not None]
    if gross_margins:
        facts.append(f"Gross Margin: {gross_margins[0]:.1%}")

    # FCF — capital allocation flexibility
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")
    if latest.operating_cash_flow is not None:
        facts.append(f"Operating Cash Flow: ${latest.operating_cash_flow / 1e9:.1f}B")

    # ROE — quality signal
    if latest.return_on_equity is not None:
        facts.append(f"ROE: {latest.return_on_equity:.1%}")

    # Valuation relative to growth
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")
        if latest.net_income and latest.net_income > 0:
            pe = market_cap / latest.net_income
            facts.append(f"P/E Ratio: {pe:.1f}")

        # PEG-like assessment
        eps_vals = [m.earnings_per_share for m in metrics if m.earnings_per_share is not None]
        if len(eps_vals) >= 2 and eps_vals[-1] and eps_vals[-1] > 0:
            shares = details.weighted_shares_outstanding or details.share_class_shares_outstanding if details else None
            if shares and shares > 0 and eps_vals[0] > 0:
                price = market_cap / shares
                pe = price / eps_vals[0]
                growth = ((eps_vals[0] / eps_vals[-1]) ** (1 / len(eps_vals)) - 1) * 100
                if growth > 0:
                    peg = pe / growth
                    facts.append(f"PEG Ratio: {peg:.2f} (P/E {pe:.1f} / Growth {growth:.0f}%)")

        if revenues:
            ps = market_cap / revenues[0]
            facts.append(f"Price/Sales: {ps:.1f}x")

    # Debt
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f}")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
