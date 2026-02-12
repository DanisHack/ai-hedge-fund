"""Benjamin Graham persona agent — deep value investing with margin of safety."""
from __future__ import annotations

import json
import logging
import math
from typing import Any

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, LLMAnalysisResult, SignalType
from src.data.polygon_client import get_company_details, get_financial_metrics
from src.graph.state import AgentState
from src.llm import call_llm

logger = logging.getLogger(__name__)

AGENT_ID = "graham_analyst"

SYSTEM_PROMPT = (
    "You are Benjamin Graham, the father of value investing and author of "
    "'The Intelligent Investor' and 'Security Analysis'. Analyze stocks using your core principles:\n\n"
    "Investment Philosophy:\n"
    "- Margin of safety: Never pay more than intrinsic value; insist on a discount\n"
    "- Financial strength: Prefer companies with current ratio >= 2.0 and low debt\n"
    "- Earnings stability: Require consistent positive earnings over multiple years\n"
    "- Graham Number: sqrt(22.5 * EPS * Book Value Per Share) as intrinsic value benchmark\n"
    "- Net-net value: Current assets minus total liabilities vs market cap for deep value\n"
    "- Conservative valuation: P/E below 15, price-to-book below 1.5\n"
    "- Defensive investing: Avoid speculation, focus on proven quantitative metrics\n\n"
    "Signal Rules:\n"
    "- BULLISH: Trading below Graham Number with adequate margin of safety AND solid financial strength\n"
    "- BEARISH: Overvalued relative to Graham Number, weak balance sheet, OR unstable earnings\n"
    "- NEUTRAL: Reasonable fundamentals but insufficient margin of safety\n\n"
    "Confidence Scale:\n"
    "- 80-100: Deep value — large margin of safety, strong balance sheet, stable earnings\n"
    "- 60-79: Good value — reasonable discount, adequate financial strength\n"
    "- 40-59: Fair value — metrics are mixed, no clear margin of safety\n"
    "- 20-39: Speculative — insufficient earnings history or weak financials\n"
    "- 0-19: Overvalued or dangerous — avoid\n\n"
    "Provide 2-4 sentences of reasoning citing specific data points. "
    "Speak in first person as Graham would, with a conservative and analytical tone."
)


def graham_agent(state: AgentState) -> dict[str, Any]:
    """Analyze stocks through Benjamin Graham's deep value investing lens.

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
    """Analyze a ticker through Graham's deep value lens using LLM reasoning."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=10)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="No financial data available for Graham-style analysis.",
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
            reasoning="Unable to complete Graham-style analysis.",
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _build_facts(metrics: list, details: Any) -> str:
    """Build structured facts string from financial data for Graham analysis."""
    latest = metrics[0]
    facts: list[str] = []

    # Shares outstanding (needed for per-share calculations)
    shares = None
    if details:
        shares = details.weighted_shares_outstanding or details.share_class_shares_outstanding

    # ── Earnings Stability ─────────────────────────────────────────
    eps_vals = [m.earnings_per_share for m in metrics if m.earnings_per_share is not None]
    if eps_vals:
        positive_eps = sum(1 for e in eps_vals if e > 0)
        facts.append(f"EPS: ${eps_vals[0]:.2f} (latest), positive in {positive_eps}/{len(eps_vals)} periods")
        if len(eps_vals) >= 2:
            eps_grew = eps_vals[0] > eps_vals[-1]
            facts.append(f"EPS Trend: ${eps_vals[-1]:.2f} → ${eps_vals[0]:.2f} "
                         f"({'growing' if eps_grew else 'declining'})")

    # Earnings consistency
    earnings = [m.net_income for m in metrics if m.net_income is not None]
    if len(earnings) >= 2:
        positive_earnings = sum(1 for e in earnings if e and e > 0)
        facts.append(f"Net Income: positive in {positive_earnings}/{len(earnings)} periods")

    # ── Financial Strength ─────────────────────────────────────────
    if latest.debt_to_equity is not None:
        facts.append(f"Debt/Equity: {latest.debt_to_equity:.2f} "
                     f"({'conservative' if latest.debt_to_equity < 0.5 else 'high' if latest.debt_to_equity > 1.0 else 'moderate'})")

    if latest.current_ratio is not None:
        facts.append(f"Current Ratio: {latest.current_ratio:.2f} "
                     f"({'strong' if latest.current_ratio >= 2.0 else 'adequate' if latest.current_ratio >= 1.5 else 'weak'})")

    # Debt ratio (total_liabilities / total_assets)
    if latest.total_assets and latest.total_liabilities and latest.total_assets > 0:
        debt_ratio = latest.total_liabilities / latest.total_assets
        facts.append(f"Debt Ratio (Liabilities/Assets): {debt_ratio:.2f} "
                     f"({'conservative' if debt_ratio < 0.5 else 'high' if debt_ratio > 0.8 else 'moderate'})")

    # ── Graham Number & Valuation ──────────────────────────────────
    bvps = None
    if details and shares and shares > 0:
        equity_vals = [m.shareholders_equity for m in metrics if m.shareholders_equity is not None]
        if equity_vals:
            bvps = equity_vals[0] / shares
            facts.append(f"Book Value Per Share: ${bvps:.2f}")

            if len(equity_vals) >= 2:
                bvps_old = equity_vals[-1] / shares
                if bvps_old > 0:
                    bv_growth = (bvps - bvps_old) / bvps_old
                    facts.append(f"Book Value Growth: {bv_growth:.1%} over {len(equity_vals)} periods")

    # Graham Number
    eps_latest = eps_vals[0] if eps_vals else None
    graham_number = None
    if eps_latest and eps_latest > 0 and bvps and bvps > 0:
        graham_number = math.sqrt(22.5 * eps_latest * bvps)
        facts.append(f"Graham Number: ${graham_number:.2f}")

    # Market cap and per-share price
    market_cap = details.market_cap if details else None
    price_per_share = None
    if market_cap and shares and shares > 0:
        price_per_share = market_cap / shares
        facts.append(f"Market Cap: ${market_cap / 1e9:.1f}B")
        facts.append(f"Price Per Share (from market cap): ${price_per_share:.2f}")

        # P/E ratio
        if eps_latest and eps_latest > 0:
            pe_ratio = price_per_share / eps_latest
            facts.append(f"P/E Ratio: {pe_ratio:.1f} "
                         f"({'below Graham threshold' if pe_ratio < 15 else 'above Graham threshold of 15'})")

        # P/B ratio
        if bvps and bvps > 0:
            pb_ratio = price_per_share / bvps
            facts.append(f"P/B Ratio: {pb_ratio:.1f} "
                         f"({'below Graham threshold' if pb_ratio < 1.5 else 'above Graham threshold of 1.5'})")

        # Margin of safety relative to Graham Number
        if graham_number and price_per_share > 0:
            margin_of_safety = (graham_number - price_per_share) / price_per_share
            facts.append(f"Margin of Safety (Graham Number): {margin_of_safety:.1%} "
                         f"({'adequate' if margin_of_safety > 0.3 else 'some' if margin_of_safety > 0 else 'negative — overvalued'})")

    # Net profit margin
    margins = [m.net_profit_margin for m in metrics if m.net_profit_margin is not None]
    if margins:
        facts.append(f"Net Margin: {margins[0]:.1%} (latest)")

    # Revenue
    revenues = [m.revenue for m in metrics if m.revenue is not None]
    if len(revenues) >= 2 and revenues[-1] and revenues[-1] > 0:
        rev_growth = (revenues[0] - revenues[-1]) / abs(revenues[-1])
        facts.append(f"Revenue: ${revenues[0] / 1e9:.1f}B (growth: {rev_growth:.1%} over {len(revenues)} periods)")

    return "\n".join(f"- {f}" for f in facts) if facts else "No financial data available."
