"""Peter Lynch persona agent — practical investor seeking ten-baggers."""
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

AGENT_ID = "lynch_analyst"

SYSTEM_PROMPT = (
    "You are Peter Lynch, the legendary manager of Fidelity Magellan Fund. "
    "Analyze stocks using your practical, common-sense investing framework:\n\n"
    "Investment Philosophy:\n"
    "- Invest in what you know: everyday businesses you can understand\n"
    "- The PEG ratio is king: P/E divided by growth rate should be below 1.0\n"
    "- Classify companies: slow growers, stalwarts, fast growers, cyclicals, turnarounds, asset plays\n"
    "- Look for 'ten-baggers': stocks that can grow 10x over time\n"
    "- Earnings growth rate is the fundamental driver of stock prices\n"
    "- Strong balance sheet (low debt) gives a company staying power\n"
    "- Favor companies with consistent, predictable earnings growth of 15-25%\n"
    "- Avoid 'whisper stocks' and hot tips — do your own homework\n"
    "- The best stocks are often boring, overlooked companies with strong fundamentals\n\n"
    "Signal Rules:\n"
    "- BULLISH: PEG < 1.0 with consistent earnings growth and manageable debt\n"
    "- BEARISH: PEG > 2.0, decelerating growth, or excessive debt\n"
    "- NEUTRAL: Fairly valued PEG (1.0-1.5) or mixed growth signals\n\n"
    "Confidence Scale:\n"
    "- 80-100: Classic ten-bagger setup — fast grower with PEG < 0.75 and clean balance sheet\n"
    "- 60-79: Good growth at a reasonable price (PEG around 1.0)\n"
    "- 40-59: Moderate growth or PEG slightly above 1.5\n"
    "- 20-39: Slow growth or high PEG ratio\n"
    "- 0-19: No growth thesis or overvalued\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Lynch would, with a folksy and practical tone."
)


def lynch_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Peter Lynch's practical GARP lens."""
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
    """Analyze a ticker through Lynch's GARP lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for GARP analysis.",
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
            reasoning="Unable to complete Lynch-style GARP analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string focused on PEG ratio and growth."""
    latest = metrics[0]
    facts: list[str] = []

    shares = None
    if details:
        shares = details.weighted_shares_outstanding or details.share_class_shares_outstanding

    # Earnings growth — the core Lynch metric
    eps_vals = [m.earnings_per_share for m in metrics if m.earnings_per_share is not None]
    if eps_vals:
        facts.append(f"EPS: ${eps_vals[0]:.2f} (latest)")
        if len(eps_vals) >= 2 and eps_vals[-1] and eps_vals[-1] > 0:
            eps_growth = (eps_vals[0] - eps_vals[-1]) / abs(eps_vals[-1])
            annual_growth_rate = (eps_vals[0] / eps_vals[-1]) ** (1 / len(eps_vals)) - 1 if eps_vals[-1] > 0 else 0
            facts.append(f"EPS Growth: {eps_growth:.1%} total, ~{annual_growth_rate:.0%} annualized")

            # PEG ratio — Lynch's signature metric
            market_cap = details.market_cap if details else None
            if market_cap and shares and shares > 0 and annual_growth_rate > 0:
                price = market_cap / shares
                pe = price / eps_vals[0] if eps_vals[0] > 0 else None
                if pe and pe > 0:
                    peg = pe / (annual_growth_rate * 100)
                    facts.append(f"P/E Ratio: {pe:.1f}")
                    peg_label = 'excellent' if peg < 0.75 else 'good' if peg < 1.0 else 'fair' if peg < 1.5 else 'expensive'
                    facts.append(f"PEG Ratio: {peg:.2f} ({peg_label})")

    # Earnings consistency
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        growing = sum(1 for i in range(len(earnings) - 1) if earnings[i] > earnings[i + 1])
        positive = sum(1 for e in earnings if e and e > 0)
        facts.append(f"Earnings: positive {positive}/{len(earnings)}, growing {growing}/{len(earnings) - 1} periods")

    # Revenue growth
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if revenues:
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B")
        if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
            rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
            facts.append(f"Revenue Growth: {rev_growth:.1%} over {len(revenues)} periods")

    # Debt — Lynch favors low debt
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f} "
                     f"({'healthy' if latest.debt_to_equity < 0.3 else 'manageable' if latest.debt_to_equity < 0.8 else 'heavy'})")

    # Net margin
    if latest.net_profit_margin is not None:
        facts.append(f"Net Margin: {latest.net_profit_margin:.1%}")

    # Free cash flow
    if latest.free_cash_flow is not None:
        facts.append(f"Free Cash Flow: ${latest.free_cash_flow / 1e9:.1f}B")

    # Market cap — Lynch likes small/mid caps
    market_cap = details.market_cap if details else None
    if market_cap:
        size = 'small cap' if market_cap < 2e9 else 'mid cap' if market_cap < 10e9 else 'large cap' if market_cap < 200e9 else 'mega cap'
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B ({size})")

    # ROE
    if latest.return_on_equity is not None:
        facts.append(f"ROE: {latest.return_on_equity:.1%}")

    # Employee count
    if details and details.total_employees:
        facts.append(f"Employees: {details.total_employees:,}")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
