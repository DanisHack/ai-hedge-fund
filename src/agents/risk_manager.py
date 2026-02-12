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
CORRELATION_LOOKBACK = 60
HIGH_CORRELATION_THRESHOLD = 0.7
CORRELATION_GROUP_CAP = 0.40


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

    # Compute pairwise correlations across all tickers + held positions
    all_tickers = set(tickers) | set(portfolio.get("positions", {}).keys())
    try:
        correlations = _compute_correlation_matrix(list(all_tickers), start_date, end_date)
        correlation_groups = _build_correlation_groups(list(all_tickers), correlations)
    except Exception:
        correlations = {}
        correlation_groups = []

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

        # 2b. Correlation adjustment (only for bullish signals)
        if consensus == "bullish" and correlation_groups:
            max_position_value, corr_reason = _correlation_adjusted_position_size(
                ticker, max_position_value, portfolio, correlation_groups,
            )
            if corr_reason:
                reasons.append(f"Correlation adjustment: {corr_reason}")

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


def _compute_correlation_matrix(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> dict[tuple[str, str], float]:
    """Compute pairwise return correlations for all tickers."""
    returns_by_ticker: dict[str, np.ndarray] = {}

    for ticker in tickers:
        try:
            prices = get_prices(ticker, start_date, end_date)
            if len(prices) >= CORRELATION_LOOKBACK:
                closes = np.array([p.close for p in prices[-CORRELATION_LOOKBACK:]])
                daily_returns = np.diff(closes) / closes[:-1]
                returns_by_ticker[ticker] = daily_returns
        except Exception:
            continue

    correlations: dict[tuple[str, str], float] = {}
    ticker_list = list(returns_by_ticker.keys())

    for i, t1 in enumerate(ticker_list):
        for j, t2 in enumerate(ticker_list):
            if i >= j:
                continue
            r1, r2 = returns_by_ticker[t1], returns_by_ticker[t2]
            min_len = min(len(r1), len(r2))
            if min_len < 20:
                continue
            corr = float(np.corrcoef(r1[:min_len], r2[:min_len])[0, 1])
            if not np.isnan(corr):
                correlations[(t1, t2)] = corr
                correlations[(t2, t1)] = corr

    return correlations


def _build_correlation_groups(
    tickers: list[str],
    correlations: dict[tuple[str, str], float],
    threshold: float = HIGH_CORRELATION_THRESHOLD,
) -> list[set[str]]:
    """Build groups of tickers with pairwise correlation above threshold.

    Uses BFS on an adjacency graph to find connected components.
    """
    adj: dict[str, set[str]] = {t: set() for t in tickers}
    for (t1, t2), corr in correlations.items():
        if corr >= threshold and t1 in adj and t2 in adj:
            adj[t1].add(t2)
            adj[t2].add(t1)

    visited: set[str] = set()
    groups: list[set[str]] = []

    for ticker in tickers:
        if ticker in visited:
            continue
        group: set[str] = set()
        queue = [ticker]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            group.add(current)
            for neighbor in adj.get(current, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        if len(group) > 1:
            groups.append(group)

    return groups


def _correlation_adjusted_position_size(
    ticker: str,
    base_max_position: float,
    portfolio: dict[str, Any],
    correlation_groups: list[set[str]],
) -> tuple[float, str | None]:
    """Reduce max_position_size if ticker is highly correlated with held positions."""
    current_positions = portfolio.get("positions", {})
    total_value = portfolio.get("total_value", 100_000)

    if not current_positions:
        return base_max_position, None

    # Find which group this ticker belongs to
    ticker_group: set[str] | None = None
    for group in correlation_groups:
        if ticker in group:
            ticker_group = group
            break

    if ticker_group is None:
        return base_max_position, None

    # Compute existing exposure in this correlation group
    group_exposure = 0.0
    correlated_held: list[str] = []
    for held_ticker, held_pos in current_positions.items():
        if held_ticker in ticker_group and held_ticker != ticker:
            held_value = held_pos.get("shares", 0) * held_pos.get("avg_cost", 0)
            group_exposure += held_value
            correlated_held.append(held_ticker)

    if not correlated_held:
        return base_max_position, None

    group_cap = total_value * CORRELATION_GROUP_CAP
    remaining_capacity = max(0, group_cap - group_exposure)
    adjusted = min(base_max_position, remaining_capacity)

    reason = (
        f"Correlated with {', '.join(correlated_held)} "
        f"(group exposure: ${group_exposure:,.0f}, "
        f"cap: ${group_cap:,.0f}, remaining: ${remaining_capacity:,.0f})"
    )

    return adjusted, reason
