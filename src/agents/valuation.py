"""Valuation analyst agent — DCF, P/E relative valuation, intrinsic value."""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, SignalType
from src.data.polygon_client import get_company_details, get_financial_metrics
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

AGENT_ID = "valuation_analyst"

# Valuation benchmarks
DISCOUNT_RATE = 0.10  # 10% WACC
TERMINAL_GROWTH = 0.03  # 3% terminal growth
DCF_PROJECTION_YEARS = 5
MARGIN_OF_SAFETY = 0.25  # 25% discount required


def valuation_agent(state: AgentState) -> dict[str, Any]:
    """Estimate intrinsic value for each ticker using DCF and relative valuation.

    Compares intrinsic value to market cap to determine if stock is
    overvalued, undervalued, or fairly valued.
    """
    data = state["data"]
    tickers: list[str] = data.get("tickers", [])
    end_date: str = data.get("end_date", "")
    show_reasoning: bool = state["metadata"].get("show_reasoning", False)

    signals: list[dict] = []

    for ticker in tickers:
        try:
            signal = _analyze_ticker(ticker, end_date)
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


def _analyze_ticker(ticker: str, end_date: str) -> AnalystSignal:
    """Run valuation analysis on a single ticker."""
    metrics = get_financial_metrics(ticker, end_date=end_date, limit=4)
    details = get_company_details(ticker)

    if not metrics:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="No financial data available for valuation.",
        )

    market_cap = details.market_cap if details else None
    latest = metrics[0]

    score = 0
    max_score = 0
    reasons: list[str] = []

    # --- 1. DCF Valuation ---
    dcf_value = _simple_dcf(metrics)
    if dcf_value is not None and market_cap is not None and market_cap > 0:
        max_score += 3
        ratio = dcf_value / market_cap

        if ratio > (1 + MARGIN_OF_SAFETY):
            score += 3
            reasons.append(f"DCF: intrinsic ${dcf_value / 1e9:.1f}B vs market ${market_cap / 1e9:.1f}B "
                           f"({ratio:.1%} of market cap, undervalued)")
        elif ratio > 0.9:
            score += 2
            reasons.append(f"DCF: intrinsic ${dcf_value / 1e9:.1f}B vs market ${market_cap / 1e9:.1f}B "
                           f"(fairly valued)")
        elif ratio > 0.6:
            score += 1
            reasons.append(f"DCF: intrinsic ${dcf_value / 1e9:.1f}B vs market ${market_cap / 1e9:.1f}B "
                           f"(moderately overvalued)")
        else:
            reasons.append(f"DCF: intrinsic ${dcf_value / 1e9:.1f}B vs market ${market_cap / 1e9:.1f}B "
                           f"(significantly overvalued)")

    # --- 2. Earnings Yield (inverse P/E) ---
    if latest.earnings_per_share and market_cap and details:
        shares = details.weighted_shares_outstanding or details.share_class_shares_outstanding
        if shares and shares > 0:
            max_score += 2
            pe_ratio = market_cap / (latest.earnings_per_share * shares)

            if 0 < pe_ratio < 15:
                score += 2
                reasons.append(f"P/E={pe_ratio:.1f} (attractively valued)")
            elif pe_ratio < 25:
                score += 1
                reasons.append(f"P/E={pe_ratio:.1f} (fairly valued)")
            elif pe_ratio > 0:
                reasons.append(f"P/E={pe_ratio:.1f} (richly valued)")
            else:
                reasons.append(f"P/E={pe_ratio:.1f} (negative earnings)")

    # --- 3. Price to Book (using equity) ---
    if latest.shareholders_equity and market_cap and latest.shareholders_equity > 0:
        max_score += 1
        pb_ratio = market_cap / latest.shareholders_equity

        if pb_ratio < 3:
            score += 1
            reasons.append(f"P/B={pb_ratio:.1f} (reasonable)")
        else:
            reasons.append(f"P/B={pb_ratio:.1f} (premium)")

    # --- 4. Free Cash Flow Yield ---
    if latest.operating_cash_flow and market_cap and market_cap > 0:
        max_score += 2
        fcf_yield = latest.operating_cash_flow / market_cap

        if fcf_yield > 0.06:
            score += 2
            reasons.append(f"FCF yield={fcf_yield:.1%} (strong)")
        elif fcf_yield > 0.03:
            score += 1
            reasons.append(f"FCF yield={fcf_yield:.1%} (moderate)")
        elif fcf_yield > 0:
            reasons.append(f"FCF yield={fcf_yield:.1%} (low)")
        else:
            reasons.append(f"FCF yield={fcf_yield:.1%} (negative cash flow)")

    # --- Determine signal ---
    if max_score == 0:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="Insufficient data for valuation.",
        )

    ratio = score / max_score
    confidence = round(ratio * 100)

    if ratio >= 0.65:
        signal = SignalType.BULLISH
    elif ratio <= 0.30:
        signal = SignalType.BEARISH
    else:
        signal = SignalType.NEUTRAL

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=signal, confidence=confidence,
        reasoning="; ".join(reasons),
    )


def _simple_dcf(metrics: list) -> Optional[float]:
    """Simple DCF using operating cash flow with growth projection.

    Uses average historical growth rate to project future cash flows,
    then discounts back to present value.
    """
    # Need at least 2 periods to estimate growth
    cash_flows = [m.operating_cash_flow for m in metrics if m.operating_cash_flow and m.operating_cash_flow > 0]

    if len(cash_flows) < 2:
        return None

    latest_cf = cash_flows[0]

    # Estimate growth from historical data
    growth_rates = []
    for i in range(len(cash_flows) - 1):
        if cash_flows[i + 1] > 0:
            rate = (cash_flows[i] - cash_flows[i + 1]) / cash_flows[i + 1]
            growth_rates.append(rate)

    if not growth_rates:
        return None

    avg_growth = sum(growth_rates) / len(growth_rates)
    # Cap growth rate at reasonable bounds
    growth_rate = max(-0.05, min(avg_growth, 0.25))

    # Project future cash flows
    projected_cf = []
    cf = latest_cf
    for year in range(1, DCF_PROJECTION_YEARS + 1):
        cf = cf * (1 + growth_rate)
        discounted = cf / (1 + DISCOUNT_RATE) ** year
        projected_cf.append(discounted)

    # Terminal value
    terminal_cf = cf * (1 + TERMINAL_GROWTH)
    terminal_value = terminal_cf / (DISCOUNT_RATE - TERMINAL_GROWTH)
    discounted_terminal = terminal_value / (1 + DISCOUNT_RATE) ** DCF_PROJECTION_YEARS

    intrinsic_value = sum(projected_cf) + discounted_terminal
    return intrinsic_value
