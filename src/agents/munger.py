"""Charlie Munger persona agent — wonderful businesses at fair prices."""
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

AGENT_ID = "munger_analyst"

SYSTEM_PROMPT = (
    "You are Charlie Munger, Warren Buffett's long-time partner at Berkshire Hathaway. "
    "Analyze stocks using your framework of quality-first investing:\n\n"
    "Investment Philosophy:\n"
    "- Only buy wonderful businesses — great economics, durable moats, pricing power\n"
    "- A great business at a fair price beats a fair business at a great price\n"
    "- Focus on return on invested capital (ROIC) — consistently high ROIC signals a moat\n"
    "- Invert, always invert: think about what can go wrong, not just what can go right\n"
    "- Mental models: use multidisciplinary thinking to understand business quality\n"
    "- Patience and discipline: sit on your hands and wait for the fat pitch\n"
    "- Avoid complexity — the best businesses are simple and understandable\n"
    "- Management must be honest and competent capital allocators\n\n"
    "Signal Rules:\n"
    "- BULLISH: Wonderful business with high ROIC, durable moat, and reasonable valuation\n"
    "- BEARISH: Mediocre business, deteriorating returns, or egregiously overvalued\n"
    "- NEUTRAL: Good business but too expensive, or insufficient evidence of moat durability\n\n"
    "Confidence Scale:\n"
    "- 80-100: Wonderful business with consistent high returns and reasonable price\n"
    "- 60-79: Good business with strong returns but valuation is not compelling\n"
    "- 40-59: Average business or unclear competitive advantage\n"
    "- 20-39: Poor economics or outside circle of competence\n"
    "- 0-19: Bad business or grossly overvalued\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Munger would, with a pithy and no-nonsense tone."
)


def munger_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Munger's quality-first investing lens."""
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
    """Analyze a ticker through Munger's quality lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for quality analysis.",
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
            reasoning="Unable to complete Munger-style quality analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on business quality and moat."""
    latest = metrics[0]
    facts: list[str] = []

    shares = None
    if details:
        shares = details.weighted_shares_outstanding or details.share_class_shares_outstanding

    # ROE consistency — the key Munger metric
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roes:
        high_roe = sum(1 for r in roes if r > 0.15)
        facts.append(f"ROE: {roes[0]:.1%} (latest), above 15% in {high_roe}/{len(roes)} periods")
        if len(roes) >= 2:
            avg_roe = sum(roes) / len(roes)
            facts.append(f"Average ROE: {avg_roe:.1%} ({'excellent' if avg_roe > 0.20 else 'good' if avg_roe > 0.15 else 'mediocre'})")

    # Profit margins — consistency is key
    net_margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if net_margins:
        facts.append(f"Net Margin: {net_margins[0]:.1%} (latest)")
        if len(net_margins) >= 2:
            margin_std = (sum((m - sum(net_margins) / len(net_margins)) ** 2 for m in net_margins) / len(net_margins)) ** 0.5
            avg_margin = sum(net_margins) / len(net_margins)
            facts.append(f"Margin Consistency: avg {avg_margin:.1%}, std dev {margin_std:.1%} "
                         f"({'stable' if margin_std < 0.03 else 'volatile'})")

    gross_margins = [m.gross_profit_margin for m in metrics if m.gross_profit_margin is not None]
    if gross_margins:
        facts.append(f"Gross Margin: {gross_margins[0]:.1%} "
                     f"({'strong pricing power' if gross_margins[0] > 0.50 else 'moderate' if gross_margins[0] > 0.30 else 'thin'})")

    # Debt — Munger hates debt
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f} "
                     f"({'conservative' if latest.debt_to_equity < 0.3 else 'moderate' if latest.debt_to_equity < 1.0 else 'aggressive'})")

    # Free cash flow per share
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")
        if shares and shares > 0:
            fcf_per_share = latest.free_cash_flow / shares
            facts.append(f"FCF Per Share: ${fcf_per_share:.2f}")

    # Earnings consistency
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        growing = sum(1 for i in range(len(earnings) - 1) if earnings[i] > earnings[i + 1])
        facts.append(f"Earnings Growth: {growing}/{len(earnings) - 1} periods growing")

    # Revenue
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B (latest)")

    # Valuation
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")
        if latest.net_income and latest.net_income > 0:
            pe = market_cap / latest.net_income
            facts.append(f"P/E Ratio: {pe:.1f}")
        if latest.free_cash_flow and latest.free_cash_flow > 0:
            fcf_yield = latest.free_cash_flow / market_cap
            facts.append(f"FCF Yield: {fcf_yield:.1%}")

    # Current ratio
    if latest.current_ratio is not None:
        facts.append(f"Current Ratio: {latest.current_ratio:.1f}")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
