"""AI Hedge Fund — Backtesting engine entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from src.backtest.engine import BacktestEngine
from src.backtest.models import BacktestResult, BenchmarkResult, PerformanceMetrics
from src.config.settings import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PROVIDER

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Hedge Fund — Backtesting Engine")
    parser.add_argument(
        "--ticker", "-t", type=str, required=True,
        help="Comma-separated stock tickers (e.g. AAPL,MSFT,NVDA)",
    )
    parser.add_argument("--start-date", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100_000, help="Starting capital (default: 100000)")
    parser.add_argument(
        "--frequency", "-f", type=str, default="weekly",
        choices=["daily", "weekly", "monthly"],
        help="Trading frequency (default: weekly)",
    )
    parser.add_argument("--lookback", type=int, default=90, help="Lookback window in days per step (default: 90)")
    parser.add_argument("--benchmark", type=str, default="SPY", help='Benchmark ticker (use "none" to disable)')
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="LLM model name")
    parser.add_argument("--provider", type=str, default=DEFAULT_MODEL_PROVIDER, help="LLM provider")
    parser.add_argument("--show-reasoning", action="store_true", help="Log agent reasoning")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


# ── Display helpers ─────────────────────────────────────────────────


def _display_summary(result: BacktestResult) -> None:
    """Display summary panel."""
    pnl = result.final_value - result.initial_cash
    pnl_pct = (pnl / result.initial_cash) * 100
    color = "green" if pnl >= 0 else "red"

    lines = [
        f"Tickers:      {', '.join(result.tickers)}",
        f"Period:       {result.start_date} to {result.end_date}",
        f"Frequency:    {result.frequency}",
        f"Steps:        {len(result.snapshots)}",
        f"",
        f"Initial:      ${result.initial_cash:>12,.2f}",
        f"Final:        ${result.final_value:>12,.2f}",
        f"P&L:          [{color}]${pnl:>12,.2f} ({pnl_pct:+.2f}%)[/{color}]",
    ]

    console.print(Panel("\n".join(lines), title="Backtest Summary", border_style="cyan"))


def _display_metrics(metrics: PerformanceMetrics, benchmark: Optional[BenchmarkResult]) -> None:
    """Display performance metrics table."""
    table = Table(title="Performance Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Strategy", justify="right")
    if benchmark:
        table.add_column(f"Benchmark ({benchmark.ticker})", justify="right")

    def _fmt(val: Optional[float], suffix: str = "%") -> str:
        if val is None:
            return "N/A"
        return f"{val:+.2f}{suffix}" if suffix == "%" else f"{val:.2f}"

    rows = [
        ("Total Return", _fmt(metrics.total_return_pct), _fmt(benchmark.total_return_pct) if benchmark else None),
        ("Annualized Return", _fmt(metrics.annualized_return_pct), _fmt(benchmark.annualized_return_pct) if benchmark else None),
        ("Sharpe Ratio", _fmt(metrics.sharpe_ratio, ""), _fmt(benchmark.sharpe_ratio, "") if benchmark else None),
        ("Max Drawdown", _fmt(metrics.max_drawdown_pct), _fmt(benchmark.max_drawdown_pct) if benchmark else None),
        ("Volatility (Annual)", _fmt(metrics.volatility_annual_pct), None),
        ("Calmar Ratio", _fmt(metrics.calmar_ratio, ""), None),
    ]

    for label, strategy_val, bench_val in rows:
        if benchmark:
            table.add_row(label, strategy_val, bench_val or "N/A")
        else:
            table.add_row(label, strategy_val)

    console.print(table)


def _display_trade_stats(metrics: PerformanceMetrics) -> None:
    """Display trade statistics table."""
    table = Table(title="Trade Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    def _fmt(val: Optional[float], suffix: str = "") -> str:
        if val is None:
            return "N/A"
        return f"{val:.2f}{suffix}"

    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row("Winning Trades", str(metrics.winning_trades))
    table.add_row("Losing Trades", str(metrics.losing_trades))
    table.add_row("Win Rate", _fmt(metrics.win_rate_pct, "%"))
    table.add_row("Avg Win", _fmt(metrics.avg_win_pct, "%"))
    table.add_row("Avg Loss", _fmt(metrics.avg_loss_pct, "%"))
    table.add_row("Profit Factor", _fmt(metrics.profit_factor))

    console.print(table)


def _display_trade_log(result: BacktestResult, max_trades: int = 20) -> None:
    """Display recent trade log."""
    trades = result.trades
    if not trades:
        console.print("[dim]No trades executed.[/dim]")
        return

    table = Table(title=f"Trade Log (last {min(max_trades, len(trades))} of {len(trades)})",
                  show_header=True, header_style="bold cyan")
    table.add_column("Date")
    table.add_column("Ticker")
    table.add_column("Action")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Value", justify="right")

    for trade in trades[-max_trades:]:
        color = "green" if trade.action == "buy" else "red"
        table.add_row(
            str(trade.date),
            trade.ticker,
            f"[{color}]{trade.action.upper()}[/{color}]",
            str(trade.quantity),
            f"${trade.price:,.2f}",
            f"${trade.total_value:,.2f}",
        )

    console.print(table)


def _display_equity_curve(result: BacktestResult, width: int = 60, height: int = 15) -> None:
    """Display ASCII equity curve."""
    snapshots = result.snapshots
    if len(snapshots) < 2:
        return

    values = [s.total_value for s in snapshots]
    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val

    if val_range == 0:
        return

    # Resample to fit width
    if len(values) > width:
        step = len(values) / width
        sampled = [values[int(i * step)] for i in range(width)]
    else:
        sampled = values
        width = len(sampled)

    # Build chart
    console.print(Panel.fit("[bold cyan]Equity Curve[/bold cyan]"))

    for row in range(height - 1, -1, -1):
        threshold = min_val + (val_range * row / (height - 1))
        # Y-axis label
        if row == height - 1:
            label = f"${max_val:>10,.0f} |"
        elif row == 0:
            label = f"${min_val:>10,.0f} |"
        elif row == height // 2:
            mid = (max_val + min_val) / 2
            label = f"${mid:>10,.0f} |"
        else:
            label = "             |"

        line_chars = []
        for v in sampled:
            if v >= threshold:
                line_chars.append("█")
            else:
                line_chars.append(" ")

        console.print(f"{label}{''.join(line_chars)}")

    # X-axis
    console.print("             +" + "─" * width)
    start_label = str(snapshots[0].date)
    end_label = str(snapshots[-1].date)
    padding = width - len(start_label) - len(end_label)
    if padding < 1:
        padding = 1
    console.print(f"              {start_label}{' ' * padding}{end_label}")


# ── Main ────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    tickers = [t.strip().upper() for t in args.ticker.split(",")]
    benchmark = args.benchmark if args.benchmark.lower() != "none" else None

    console.print(f"\n[bold green]AI Hedge Fund — Backtester[/bold green]")
    console.print(f"Tickers: {', '.join(tickers)} | Period: {args.start_date} to {args.end_date} | Freq: {args.frequency}\n")

    engine = BacktestEngine(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_cash=args.cash,
        frequency=args.frequency,
        lookback_days=args.lookback,
        model_name=args.model,
        model_provider=args.provider,
        show_reasoning=args.show_reasoning,
        benchmark_ticker=benchmark,
    )

    try:
        result = engine.run()
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    console.print()
    _display_summary(result)
    console.print()
    _display_metrics(result.metrics, result.benchmark)
    console.print()
    _display_trade_stats(result.metrics)
    console.print()
    _display_trade_log(result)
    console.print()
    _display_equity_curve(result)
    console.print()


if __name__ == "__main__":
    main()
