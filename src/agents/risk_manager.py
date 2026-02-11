"""Risk manager agent (rule-based)."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage

from src.data.polygon_client import get_prices
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

AGENT_ID = "risk_manager"

MAX_POSITION_PCT = 0.25
MAX_TOTAL_EXPOSURE_PCT = 0.90
VOLATILITY_LOOKBACK = 20
HIGH_VOLATILITY_THRESHOLD = 0.03


def risk_manager_agent(state: AgentState) -> dict[str, Any]:
    """Review analyst signals and apply risk constraints.

    Adjusts confidence based on: volatility, concentration, portfolio exposure.
    """
    data = state["data"]
    tickers = data.get("tickers", [])
    analyst_signals = data.get("analyst_signals", {})
    portfolio = data.get("portfolio", {"cash": 100000, "positions": {}, "total_value": 100000})
    start_date = data.get("start_date", "")
    end_date = data.get("end_date", "")
    show_reasoning = state["metadata"].get("show_reasoning", False)

    risk_adjusted: list[dict] = []

    for ticker in tickers:
        ticker_signals = _collect_signals_for_ticker(ticker, analyst_signals)

        if not ticker_signals:
            risk_adjusted.append({
                "ticker": ticker, "signal": "neutral",
                "confidence": 0, "reasoning": "No analyst signals received.",
                "max_position_size": 0,
            })
            continue

        # Aggregate consensus
        avg_confidence = sum(s["confidence"] for s in ticker_signals) / len(ticker_signals)
        bullish = sum(1 for s in ticker_signals if s["signal"] == "bullish")
        bearish = sum(1 for s in ticker_signals if s["signal"] == "bearish")

        if bullish > bearish:
            consensus = "bullish"
        elif bearish > bullish:
            consensus = "bearish"
        else:
            consensus = "neutral"

        adjusted_confidence = avg_confidence
        reasons: list[str] = []

        # 1. Volatility check
        try:
            prices = get_prices(ticker, start_date, end_date)
            if len(prices) >= VOLATILITY_LOOKBACK:
                closes = np.array([p.close for p in prices[-VOLATILITY_LOOKBACK:]])
                daily_returns = np.diff(closes) / closes[:-1]
                volatility = float(np.std(daily_returns))
                if volatility > HIGH_VOLATILITY_THRESHOLD:
                    penalty = min(30, (volatility - HIGH_VOLATILITY_THRESHOLD) * 1000)
                    adjusted_confidence = max(0, adjusted_confidence - penalty)
                    reasons.append(f"High volatility ({volatility:.1%} daily), confidence -{penalty:.0f}")
        except Exception:
            reasons.append("Could not compute volatility")

        # 2. Position concentration limit
        total_value = portfolio.get("total_value", 100000)
        max_position_value = total_value * MAX_POSITION_PCT
        reasons.append(f"Max position: ${max_position_value:,.0f} ({MAX_POSITION_PCT:.0%} of portfolio)")

        # 3. Overall exposure
        current_positions = portfolio.get("positions", {})
        invested = sum(pos.get("value", 0) for pos in current_positions.values())
        exposure_pct = invested / total_value if total_value > 0 else 0
        remaining_capacity = max(0, MAX_TOTAL_EXPOSURE_PCT - exposure_pct)
        if remaining_capacity <= 0 and consensus == "bullish":
            adjusted_confidence = max(0, adjusted_confidence - 50)
            reasons.append(f"Portfolio near max exposure ({exposure_pct:.0%})")

        risk_adjusted.append({
            "ticker": ticker,
            "signal": consensus,
            "confidence": round(adjusted_confidence),
            "reasoning": "; ".join(reasons) if reasons else "No risk flags",
            "max_position_size": max_position_value,
        })

        if show_reasoning:
            logger.info(f"[{AGENT_ID}] {ticker}: {consensus} "
                        f"(adjusted confidence={adjusted_confidence:.0f})")

    message = HumanMessage(
        content=json.dumps({"agent": AGENT_ID, "signals": risk_adjusted}),
        name=AGENT_ID,
    )
    return {
        "messages": [message],
        "data": {"risk_adjusted_signals": risk_adjusted},
    }


def _collect_signals_for_ticker(ticker: str, analyst_signals: dict) -> list[dict]:
    """Gather all analyst signals for a given ticker."""
    result = []
    for _agent_id, signals in analyst_signals.items():
        for sig in signals:
            if sig.get("ticker") == ticker:
                result.append(sig)
    return result
