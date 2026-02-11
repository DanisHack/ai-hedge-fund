"""Performance metrics and benchmark computation for backtesting."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

from src.backtest.models import (
    BenchmarkResult,
    PerformanceMetrics,
    PortfolioSnapshot,
    Trade,
)
from src.data.polygon_client import get_prices

logger = logging.getLogger(__name__)

RISK_FREE_RATE = 0.04  # 4% annual


def compute_metrics(
    snapshots: list[PortfolioSnapshot],
    trades: list[Trade],
    initial_cash: float,
) -> PerformanceMetrics:
    """Compute performance metrics from backtest snapshots and trades."""
    if not snapshots:
        return PerformanceMetrics(
            total_return_pct=0.0,
            annualized_return_pct=0.0,
            max_drawdown_pct=0.0,
            total_trades=0,
        )

    final_value = snapshots[-1].total_value
    total_return = (final_value - initial_cash) / initial_cash

    # Days in backtest
    days = (snapshots[-1].date - snapshots[0].date).days
    if days <= 0:
        days = 1

    # Annualized return
    annualized_return = (1 + total_return) ** (365 / days) - 1

    # Daily returns array
    daily_returns = np.array([
        s.daily_return for s in snapshots if s.daily_return is not None
    ])

    # Volatility (annualized)
    volatility = None
    if len(daily_returns) > 1:
        volatility = float(np.std(daily_returns, ddof=1) * np.sqrt(252))

    # Sharpe ratio
    sharpe = None
    if volatility is not None and volatility > 0:
        sharpe = (annualized_return - RISK_FREE_RATE) / volatility

    # Max drawdown
    values = np.array([s.total_value for s in snapshots])
    cummax = np.maximum.accumulate(values)
    drawdowns = (values - cummax) / cummax
    max_dd = float(np.min(drawdowns))

    dd_end_idx = int(np.argmin(drawdowns))
    dd_start_idx = int(np.argmax(values[:dd_end_idx + 1])) if dd_end_idx > 0 else 0
    max_dd_start = snapshots[dd_start_idx].date
    max_dd_end = snapshots[dd_end_idx].date

    # Calmar ratio
    calmar = None
    if max_dd != 0:
        calmar = annualized_return / abs(max_dd)

    # Win/loss analysis via FIFO matching of buy→sell pairs per ticker
    win_rate, winning, losing, avg_win, avg_loss, profit_factor = _analyze_trades(trades)
    total_trades = winning + losing

    return PerformanceMetrics(
        total_return_pct=round(total_return * 100, 2),
        annualized_return_pct=round(annualized_return * 100, 2),
        sharpe_ratio=round(sharpe, 2) if sharpe is not None else None,
        max_drawdown_pct=round(max_dd * 100, 2),
        max_drawdown_start=max_dd_start,
        max_drawdown_end=max_dd_end,
        win_rate_pct=round(win_rate, 2) if win_rate is not None else None,
        total_trades=total_trades,
        winning_trades=winning,
        losing_trades=losing,
        avg_win_pct=round(avg_win, 2) if avg_win is not None else None,
        avg_loss_pct=round(avg_loss, 2) if avg_loss is not None else None,
        profit_factor=round(profit_factor, 2) if profit_factor is not None else None,
        volatility_annual_pct=round(volatility * 100, 2) if volatility is not None else None,
        calmar_ratio=round(calmar, 2) if calmar is not None else None,
    )


def _analyze_trades(
    trades: list[Trade],
) -> tuple[
    Optional[float],  # win_rate
    int,  # winning
    int,  # losing
    Optional[float],  # avg_win_pct
    Optional[float],  # avg_loss_pct
    Optional[float],  # profit_factor
]:
    """FIFO matching of buy→sell pairs per ticker to compute win/loss stats."""
    # Build FIFO queues of buys per ticker
    buy_queue: dict[str, list[tuple[int, float]]] = defaultdict(list)  # (qty, price)
    round_trips: list[float] = []  # pct return per round trip

    for trade in sorted(trades, key=lambda t: t.date):
        if trade.action == "buy":
            buy_queue[trade.ticker].append((trade.quantity, trade.price))
        elif trade.action == "sell":
            remaining = trade.quantity
            while remaining > 0 and buy_queue[trade.ticker]:
                buy_qty, buy_price = buy_queue[trade.ticker][0]
                matched = min(remaining, buy_qty)
                if buy_price > 0:
                    pct_return = ((trade.price - buy_price) / buy_price) * 100
                    round_trips.append(pct_return)
                remaining -= matched
                buy_qty -= matched
                if buy_qty <= 0:
                    buy_queue[trade.ticker].pop(0)
                else:
                    buy_queue[trade.ticker][0] = (buy_qty, buy_price)

    if not round_trips:
        return None, 0, 0, None, None, None

    wins = [r for r in round_trips if r > 0]
    losses = [r for r in round_trips if r <= 0]

    winning = len(wins)
    losing = len(losses)
    win_rate = (winning / len(round_trips)) * 100

    avg_win = float(np.mean(wins)) if wins else None
    avg_loss = float(np.mean(losses)) if losses else None

    total_gains = sum(wins)
    total_losses = abs(sum(losses))
    profit_factor = total_gains / total_losses if total_losses > 0 else None

    return win_rate, winning, losing, avg_win, avg_loss, profit_factor


def compute_benchmark(
    ticker: str,
    start_date: str,
    end_date: str,
    initial_cash: float,
) -> Optional[BenchmarkResult]:
    """Simulate buy-and-hold for a benchmark ticker."""
    try:
        prices = get_prices(ticker, start_date, end_date)
        if len(prices) < 2:
            logger.warning(f"Insufficient price data for benchmark {ticker}")
            return None

        start_price = prices[0].close
        end_price = prices[-1].close

        # Buy max shares on day 1
        shares = int(initial_cash / start_price)
        if shares <= 0:
            return None
        leftover_cash = initial_cash - (shares * start_price)

        # Daily portfolio values for metrics
        daily_values = []
        for p in prices:
            daily_values.append(shares * p.close + leftover_cash)

        values = np.array(daily_values)
        final_value = values[-1]
        total_return = (final_value - initial_cash) / initial_cash

        days = (prices[-1].timestamp.date() - prices[0].timestamp.date()).days
        if days <= 0:
            days = 1
        annualized = (1 + total_return) ** (365 / days) - 1

        # Daily returns
        daily_returns = np.diff(values) / values[:-1]
        vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else None
        sharpe = (annualized - RISK_FREE_RATE) / vol if vol and vol > 0 else None

        # Max drawdown
        cummax = np.maximum.accumulate(values)
        drawdowns = (values - cummax) / cummax
        max_dd = float(np.min(drawdowns))

        return BenchmarkResult(
            ticker=ticker,
            start_price=round(start_price, 2),
            end_price=round(end_price, 2),
            total_return_pct=round(total_return * 100, 2),
            annualized_return_pct=round(annualized * 100, 2),
            sharpe_ratio=round(sharpe, 2) if sharpe is not None else None,
            max_drawdown_pct=round(max_dd * 100, 2),
        )
    except Exception as e:
        logger.warning(f"Failed to compute benchmark for {ticker}: {e}")
        return None
