"""Portfolio manager agent (rule-based)."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.data.models import Portfolio, Position, TradeAction
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

AGENT_ID = "portfolio_manager"

BUY_CONFIDENCE_THRESHOLD = 50
SELL_CONFIDENCE_THRESHOLD = 50
MIN_TRADE_VALUE = 100


def portfolio_manager_agent(state: AgentState) -> dict[str, Any]:
    """Convert risk-adjusted signals into trade actions (buy/sell/hold)."""
    data = state["data"]
    risk_signals = data.get("risk_adjusted_signals", [])
    current_prices = data.get("current_prices", {})
    portfolio_state = data.get("portfolio", {"cash": 100000, "positions": {}, "total_value": 100000})
    show_reasoning = state["metadata"].get("show_reasoning", False)

    cash = portfolio_state.get("cash", 100000)
    positions: list[dict] = []

    for signal in risk_signals:
        ticker = signal["ticker"]
        direction = signal["signal"]
        confidence = signal["confidence"]
        max_position = signal.get("max_position_size", cash)
        price = current_prices.get(ticker)

        if price is None or price <= 0:
            positions.append(Position(
                ticker=ticker, action=TradeAction.HOLD,
                quantity=0, confidence=confidence,
                reasoning=f"No price available for {ticker}",
            ).model_dump(mode="json"))
            continue

        reasoning_parts = [f"Signal: {direction} @ {confidence}% confidence"]

        if direction == "bullish" and confidence >= BUY_CONFIDENCE_THRESHOLD:
            allocation_pct = min(confidence / 100, 1.0) * 0.5
            trade_value = min(max_position * allocation_pct, cash)

            if trade_value < MIN_TRADE_VALUE:
                positions.append(Position(
                    ticker=ticker, action=TradeAction.HOLD,
                    quantity=0, confidence=confidence,
                    reasoning=f"Insufficient cash for meaningful position (${trade_value:.0f})",
                ).model_dump(mode="json"))
                continue

            quantity = int(trade_value / price)
            if quantity <= 0:
                action = TradeAction.HOLD
                reasoning_parts.append(f"Price ${price:.2f} too high for allocation ${trade_value:.0f}")
            else:
                action = TradeAction.BUY
                cash -= quantity * price
                reasoning_parts.append(f"Buy {quantity} shares @ ${price:.2f} = ${quantity * price:,.0f}")

            positions.append(Position(
                ticker=ticker, action=action,
                quantity=quantity, confidence=confidence,
                reasoning="; ".join(reasoning_parts),
            ).model_dump(mode="json"))

        elif direction == "bearish" and confidence >= SELL_CONFIDENCE_THRESHOLD:
            existing = portfolio_state.get("positions", {}).get(ticker, {})
            existing_qty = existing.get("shares", 0)

            if existing_qty > 0:
                action = TradeAction.SELL
                cash += existing_qty * price
                reasoning_parts.append(f"Sell all {existing_qty} shares @ ${price:.2f}")
            else:
                action = TradeAction.HOLD
                existing_qty = 0
                reasoning_parts.append("No position to sell")

            positions.append(Position(
                ticker=ticker, action=action,
                quantity=existing_qty, confidence=confidence,
                reasoning="; ".join(reasoning_parts),
            ).model_dump(mode="json"))

        else:
            positions.append(Position(
                ticker=ticker, action=TradeAction.HOLD,
                quantity=0, confidence=confidence,
                reasoning=f"Below threshold (need {BUY_CONFIDENCE_THRESHOLD}% for buy, "
                          f"{SELL_CONFIDENCE_THRESHOLD}% for sell)",
            ).model_dump(mode="json"))

    portfolio_output = Portfolio(
        positions=[Position(**p) for p in positions],
        cash_remaining=cash,
        total_value=cash + sum(
            p.get("quantity", 0) * current_prices.get(p["ticker"], 0)
            for p in positions if p.get("action") == "buy"
        ),
    ).model_dump(mode="json")

    if show_reasoning:
        for p in positions:
            logger.info(f"[{AGENT_ID}] {p['ticker']}: {p['action']} "
                        f"{p['quantity']} (confidence={p['confidence']})")

    message = HumanMessage(
        content=json.dumps({"agent": AGENT_ID, "portfolio": portfolio_output}),
        name=AGENT_ID,
    )
    return {
        "messages": [message],
        "data": {"portfolio_output": portfolio_output},
    }
