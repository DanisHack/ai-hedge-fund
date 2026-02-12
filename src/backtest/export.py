"""Export backtest results to JSON or CSV."""
from __future__ import annotations

import csv
import json
import os
from datetime import date, datetime
from typing import Any

from src.backtest.models import BacktestResult


def _json_serial(obj: Any) -> Any:
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def export_json(result: BacktestResult, path: str) -> None:
    """Export BacktestResult to a JSON file."""
    data = result.model_dump(mode="json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_serial)


def export_csv(result: BacktestResult, path: str) -> None:
    """Export BacktestResult to a multi-section CSV file."""
    metrics = result.metrics
    benchmark = result.benchmark

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        # Section 1: Summary
        writer.writerow(["# Summary"])
        summary_headers = [
            "tickers", "start_date", "end_date", "frequency",
            "initial_cash", "final_value", "total_return_pct",
        ]
        writer.writerow(summary_headers)
        writer.writerow([
            ";".join(result.tickers),
            str(result.start_date),
            str(result.end_date),
            result.frequency,
            result.initial_cash,
            result.final_value,
            metrics.total_return_pct,
        ])
        writer.writerow([])

        # Section 2: Metrics
        writer.writerow(["# Metrics"])
        metrics_headers = [
            "total_return_pct", "annualized_return_pct", "sharpe_ratio",
            "max_drawdown_pct", "volatility_annual_pct", "calmar_ratio",
            "total_trades", "winning_trades", "losing_trades",
            "win_rate_pct", "avg_win_pct", "avg_loss_pct", "profit_factor",
        ]
        if benchmark:
            metrics_headers += [
                "benchmark_ticker", "benchmark_total_return_pct",
                "benchmark_annualized_return_pct", "benchmark_sharpe_ratio",
                "benchmark_max_drawdown_pct",
            ]
        writer.writerow(metrics_headers)

        def _v(val: Any) -> str:
            return "" if val is None else str(val)

        metrics_row: list[Any] = [
            _v(metrics.total_return_pct),
            _v(metrics.annualized_return_pct),
            _v(metrics.sharpe_ratio),
            _v(metrics.max_drawdown_pct),
            _v(metrics.volatility_annual_pct),
            _v(metrics.calmar_ratio),
            metrics.total_trades,
            metrics.winning_trades,
            metrics.losing_trades,
            _v(metrics.win_rate_pct),
            _v(metrics.avg_win_pct),
            _v(metrics.avg_loss_pct),
            _v(metrics.profit_factor),
        ]
        if benchmark:
            metrics_row += [
                benchmark.ticker,
                _v(benchmark.total_return_pct),
                _v(benchmark.annualized_return_pct),
                _v(benchmark.sharpe_ratio),
                _v(benchmark.max_drawdown_pct),
            ]
        writer.writerow(metrics_row)
        writer.writerow([])

        # Section 3: Trades
        writer.writerow(["# Trades"])
        trade_headers = ["date", "ticker", "action", "quantity", "price", "total_value"]
        writer.writerow(trade_headers)
        for trade in result.trades:
            writer.writerow([
                str(trade.date), trade.ticker, trade.action,
                trade.quantity, trade.price, trade.total_value,
            ])
        writer.writerow([])

        # Section 4: Snapshots
        writer.writerow(["# Snapshots"])
        ticker_cols = [f"holdings_{t}" for t in result.tickers]
        snapshot_headers = ["date", "cash", "total_value", "daily_return"] + ticker_cols
        writer.writerow(snapshot_headers)
        for snap in result.snapshots:
            row: list[Any] = [
                str(snap.date),
                snap.cash,
                snap.total_value,
                _v(snap.daily_return),
            ]
            for t in result.tickers:
                holding = snap.holdings.get(t)
                row.append(holding.shares if holding else 0)
            writer.writerow(row)


def export_results(result: BacktestResult, path: str) -> None:
    """Export results to JSON or CSV based on file extension.

    Raises ValueError for unsupported extensions.
    """
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".json":
        export_json(result, path)
    elif ext == ".csv":
        export_csv(result, path)
    else:
        raise ValueError(f"Unsupported export format: '{ext}'. Use .json or .csv")
