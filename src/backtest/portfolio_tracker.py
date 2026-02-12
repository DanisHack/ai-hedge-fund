"""Running portfolio state and trade execution for backtesting."""
from __future__ import annotations

import logging
from datetime import date
from typing import Any

from src.backtest.models import HoldingDetail, PortfolioSnapshot, Trade

logger = logging.getLogger(__name__)


class PortfolioTracker:
    """Source of truth for cash and positions across backtest iterations."""

    def __init__(self, initial_cash: float, commission_rate: float = 0.0, slippage_rate: float = 0.0) -> None:
        self.cash: float = initial_cash
        self.initial_cash: float = initial_cash
        self.commission_rate: float = commission_rate
        self.slippage_rate: float = slippage_rate
        # {ticker: {"shares": int, "avg_cost": float}}
        self.positions: dict[str, dict[str, Any]] = {}
        self.snapshots: list[PortfolioSnapshot] = []
        self.trades: list[Trade] = []

    def get_portfolio_dict(self) -> dict[str, Any]:
        """Return portfolio state in the format expected by run_hedge_fund()."""
        total_value = self.cash
        positions_out: dict[str, dict[str, Any]] = {}
        for ticker, pos in self.positions.items():
            positions_out[ticker] = {
                "shares": pos["shares"],
                "avg_cost": pos["avg_cost"],
            }
            # Note: total_value will be approximate here since we don't have
            # current prices; the workflow will use its own price data.
            total_value += pos["shares"] * pos["avg_cost"]

        return {
            "cash": self.cash,
            "positions": positions_out,
            "total_value": total_value,
        }

    def apply_trades(
        self,
        portfolio_output: dict[str, Any],
        current_prices: dict[str, float],
        trade_date: date,
    ) -> None:
        """Read positions from workflow output and execute buys/sells against tracker state."""
        positions = portfolio_output.get("positions", [])

        for pos in positions:
            ticker = pos.get("ticker", "")
            action = pos.get("action", "hold")
            quantity = pos.get("quantity", 0)
            price = current_prices.get(ticker)

            if price is None or price <= 0 or quantity <= 0:
                continue

            if action == "buy":
                effective_price = price * (1 + self.commission_rate + self.slippage_rate)
                cost = quantity * price
                total_cost = quantity * effective_price
                # If insufficient cash, buy what we can afford
                if total_cost > self.cash:
                    quantity = int(self.cash / effective_price)
                    if quantity <= 0:
                        continue
                    cost = quantity * price
                    total_cost = quantity * effective_price

                commission = cost * self.commission_rate
                slippage_cost = cost * self.slippage_rate
                self.cash -= total_cost

                if ticker in self.positions:
                    existing = self.positions[ticker]
                    total_shares = existing["shares"] + quantity
                    # Weighted average cost
                    existing["avg_cost"] = (
                        (existing["shares"] * existing["avg_cost"]) + cost
                    ) / total_shares
                    existing["shares"] = total_shares
                else:
                    self.positions[ticker] = {
                        "shares": quantity,
                        "avg_cost": price,
                    }

                self.trades.append(Trade(
                    date=trade_date,
                    ticker=ticker,
                    action="buy",
                    quantity=quantity,
                    price=price,
                    total_value=cost,
                    commission=commission,
                    slippage=slippage_cost,
                ))

            elif action == "sell":
                if ticker not in self.positions:
                    continue
                existing = self.positions[ticker]
                # Sell min(requested, held)
                sell_qty = min(quantity, existing["shares"])
                if sell_qty <= 0:
                    continue

                gross_proceeds = sell_qty * price
                commission = gross_proceeds * self.commission_rate
                slippage_cost = gross_proceeds * self.slippage_rate
                net_proceeds = gross_proceeds - commission - slippage_cost
                self.cash += net_proceeds
                existing["shares"] -= sell_qty

                if existing["shares"] <= 0:
                    del self.positions[ticker]

                self.trades.append(Trade(
                    date=trade_date,
                    ticker=ticker,
                    action="sell",
                    quantity=sell_qty,
                    price=price,
                    total_value=gross_proceeds,
                    commission=commission,
                    slippage=slippage_cost,
                ))

    def take_snapshot(self, snap_date: date, current_prices: dict[str, float]) -> PortfolioSnapshot:
        """Record portfolio value and compute daily return vs previous snapshot."""
        holdings: dict[str, HoldingDetail] = {}
        holdings_value = 0.0

        for ticker, pos in self.positions.items():
            price = current_prices.get(ticker, pos["avg_cost"])
            market_value = pos["shares"] * price
            holdings_value += market_value
            holdings[ticker] = HoldingDetail(
                shares=pos["shares"],
                avg_cost=pos["avg_cost"],
                current_price=price,
                market_value=market_value,
                unrealized_pnl=market_value - (pos["shares"] * pos["avg_cost"]),
            )

        total_value = self.cash + holdings_value

        daily_return = None
        if self.snapshots:
            prev_value = self.snapshots[-1].total_value
            if prev_value > 0:
                daily_return = (total_value - prev_value) / prev_value

        snapshot = PortfolioSnapshot(
            date=snap_date,
            cash=self.cash,
            holdings=holdings,
            total_value=total_value,
            daily_return=daily_return,
        )
        self.snapshots.append(snapshot)
        return snapshot
