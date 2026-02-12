"""Phil Fisher persona agent — meticulous growth investor using scuttlebutt research."""
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

AGENT_ID = "fisher_analyst"

SYSTEM_PROMPT = (
    "You are Phil Fisher, the pioneering growth investor and author of "
    "'Common Stocks and Uncommon Profits'. Analyze stocks using your growth framework:\n\n"
    "Investment Philosophy:\n"
    "- Invest in companies with above-average growth potential for years to come\n"
    "- 'Scuttlebutt' research: understand the company deeply through its products, competition, and management\n"
    "- Revenue growth driven by R&D and innovation — not just cost-cutting\n"
    "- Profit margins should be expanding or sustainably high\n"
    "- Management integrity and long-term vision matter enormously\n"
    "- Hold for the long term: the best time to sell is almost never\n"
    "- Pay a fair price for outstanding companies — don't wait for a bargain on a great business\n"
    "- Concentration: a few excellent companies beats diversification across mediocre ones\n"
    "- Look for companies reinvesting in R&D and sales capacity\n\n"
    "Signal Rules:\n"
    "- BULLISH: Strong revenue growth, expanding margins, heavy R&D reinvestment, capable management\n"
    "- BEARISH: Stagnant revenue, contracting margins, or undisciplined spending\n"
    "- NEUTRAL: Some growth but unclear competitive advantage or margin trajectory\n\n"
    "Confidence Scale:\n"
    "- 80-100: Outstanding growth company with expanding margins and strong reinvestment\n"
    "- 60-79: Good growth with stable margins and reasonable reinvestment\n"
    "- 40-59: Moderate growth or unclear margin direction\n"
    "- 20-39: Weak growth or deteriorating fundamentals\n"
    "- 0-19: No growth thesis or poor management indicators\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Fisher would, with a thoughtful and research-focused tone."
)


def fisher_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Phil Fisher's growth investing lens."""
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
    """Analyze a ticker through Fisher's growth lens using LLM reasoning."""
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
            reasoning="Unable to complete Fisher-style growth analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on growth quality and reinvestment."""
    latest = metrics[0]
    facts: list[str] = []

    # Revenue growth trajectory — Fisher's primary focus
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B (latest)")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            facts.append(f"Revenue Growth: {rev_growth:.1%} over {len(revenues)} periods")

            # Growth consistency
            growing = sum(1 for i in range(len(revenues) - 1) if revenues[i] > revenues[i + 1])
            facts.append(f"Revenue Growing: {growing}/{len(revenues) - 1} consecutive periods")

    # Margin expansion — key Fisher signal
    gross_margins = [m.gross_profit_margin for m in metrics if m.gross_profit_margin is not None]
    if gross_margins:
        facts.append(f"Gross Margin: {gross_margins[0]:.1%} (latest)")
        if len(gross_margins) >= 2:
            trend = "EXPANDING" if gross_margins[0] > gross_margins[-1] else "contracting"
            facts.append(f"Gross Margin Trend: {gross_margins[-1]:.1%} → {gross_margins[0]:.1%} ({trend})")

    net_margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if net_margins:
        facts.append(f"Net Margin: {net_margins[0]:.1%} (latest)")
        if len(net_margins) >= 2:
            trend = "EXPANDING" if net_margins[0] > net_margins[-1] else "contracting"
            facts.append(f"Net Margin Trend: {net_margins[-1]:.1%} → {net_margins[0]:.1%} ({trend})")

    # R&D and reinvestment proxy (capex = OCF - FCF)
    if latest.operating_cash_flow and latest.free_cash_flow and latest.revenue:
        capex = latest.operating_cash_flow - latest.free_cash_flow
        reinvestment_rate = capex / latest.revenue
        facts.append(f"Reinvestment Rate (capex/revenue): {reinvestment_rate:.1%} "
                     f"({'heavy R&D/capex' if reinvestment_rate > 0.12 else 'moderate' if reinvestment_rate > 0.05 else 'light'})")

    # Earnings growth
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        if earnings[-1] and earnings[-1] > 0:
            earn_growth = (earnings[0] - earnings[-1]) / abs(earnings[-1])
            facts.append(f"Earnings Growth: {earn_growth:.1%} over {len(earnings)} periods")

    # ROE — profitability quality
    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if roes:
        facts.append(f"ROE: {roes[0]:.1%} (latest)")
        if len(roes) >= 2:
            avg_roe = sum(roes) / len(roes)
            facts.append(f"Average ROE: {avg_roe:.1%}")

    # Free cash flow
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")

    # Debt — Fisher prefers low debt
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f}")

    # Market cap and valuation
    market_cap = details.market_cap if details else None
    if market_cap:
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")
        if revenues:
            ps = market_cap / revenues[0]
            facts.append(f"Price/Sales: {ps:.1f}x")
        if latest.net_income and latest.net_income > 0:
            pe = market_cap / latest.net_income
            facts.append(f"P/E Ratio: {pe:.1f}")

    # Employees
    if details and details.total_employees:
        facts.append(f"Employees: {details.total_employees:,}")
        if revenues:
            rev_per_employee = revenues[0] / details.total_employees
            facts.append(f"Revenue Per Employee: ${rev_per_employee / 1000:.0f}K")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
