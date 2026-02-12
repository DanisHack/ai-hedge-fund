"""AI Hedge Fund — Paper Trading CLI entry point."""
from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.backtest.metrics import compute_metrics
from src.config.settings import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PROVIDER
from src.paper_trading.runner import PaperTradingRunner
from src.paper_trading.state import (
    PaperTradingConfig,
    PaperTradingState,
    create_initial_state,
    load_state,
    save_state,
)

console = Console()

DEFAULT_STATE_FILE = "paper_portfolio.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Hedge Fund — Paper Trading")
    subparsers = parser.add_subparsers(dest="command", help="Paper trading commands")

    # ── Shared arguments ───────────────────────────────────────────
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--state-file", type=str, default=DEFAULT_STATE_FILE,
                        help=f"Path to state file (default: {DEFAULT_STATE_FILE})")

    # ── run ─────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", parents=[shared],
                                       help="Run one trading cycle")
    run_parser.add_argument("--ticker", "-t", type=str, required=True,
                            help="Comma-separated stock tickers (e.g. AAPL,MSFT,NVDA)")
    run_parser.add_argument("--cash", type=float, default=100_000,
                            help="Starting capital for new portfolios (default: 100000)")
    run_parser.add_argument("--lookback", type=int, default=90,
                            help="Lookback window in days (default: 90)")
    run_parser.add_argument("--commission", type=float, default=0.001,
                            help="Commission rate per trade (default: 0.001)")
    run_parser.add_argument("--slippage", type=float, default=0.00005,
                            help="Slippage rate per trade (default: 0.00005)")
    run_parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME,
                            help="LLM model name")
    run_parser.add_argument("--provider", type=str, default=DEFAULT_MODEL_PROVIDER,
                            help="LLM provider")
    run_parser.add_argument("--use-llm", action="store_true", default=False,
                            help="Use LLM reasoning for analyst agents")
    run_parser.add_argument("--personas", type=str, default=None,
                            help='Investor personas (e.g. buffett,graham or "all")')
    run_parser.add_argument("--stop-loss", type=float, default=None,
                            help="Fixed stop-loss percentage as decimal (e.g. 0.10 for 10%%)")
    run_parser.add_argument("--trailing-stop", type=float, default=None,
                            help="Trailing stop-loss percentage as decimal (e.g. 0.10 for 10%%)")
    run_parser.add_argument("--take-profit", type=float, default=None,
                            help="Take-profit percentage as decimal (e.g. 0.20 for 20%%)")
    run_parser.add_argument("--show-reasoning", action="store_true",
                            help="Log agent reasoning")
    run_parser.add_argument("--debug", action="store_true",
                            help="Enable debug logging")

    # ── status ──────────────────────────────────────────────────────
    subparsers.add_parser("status", parents=[shared],
                          help="Show current portfolio without trading")

    # ── reset ───────────────────────────────────────────────────────
    reset_parser = subparsers.add_parser("reset", parents=[shared],
                                          help="Reset portfolio to fresh state")
    reset_parser.add_argument("--ticker", "-t", type=str, required=True,
                              help="Comma-separated stock tickers")
    reset_parser.add_argument("--cash", type=float, default=100_000,
                              help="Starting capital (default: 100000)")
    reset_parser.add_argument("--lookback", type=int, default=90,
                              help="Lookback window in days (default: 90)")
    reset_parser.add_argument("--commission", type=float, default=0.001,
                              help="Commission rate per trade (default: 0.001)")
    reset_parser.add_argument("--slippage", type=float, default=0.00005,
                              help="Slippage rate per trade (default: 0.00005)")
    reset_parser.add_argument("--stop-loss", type=float, default=None,
                              help="Fixed stop-loss percentage as decimal (e.g. 0.10 for 10%%)")
    reset_parser.add_argument("--trailing-stop", type=float, default=None,
                              help="Trailing stop-loss percentage as decimal (e.g. 0.10 for 10%%)")
    reset_parser.add_argument("--take-profit", type=float, default=None,
                              help="Take-profit percentage as decimal (e.g. 0.20 for 20%%)")

    return parser.parse_args()


# ── Display helpers ─────────────────────────────────────────────────


def _display_portfolio(state: PaperTradingState) -> None:
    """Display portfolio summary panel."""
    portfolio = state.portfolio
    holdings_value = sum(
        pos.shares * pos.avg_cost for pos in portfolio.positions.values()
    )
    total_value = portfolio.cash + holdings_value

    # Use latest snapshot for more accurate valuation if available
    if state.snapshots:
        total_value = state.snapshots[-1].total_value
        holdings_value = total_value - portfolio.cash

    pnl = total_value - state.config.initial_cash
    pnl_pct = (pnl / state.config.initial_cash) * 100
    color = "green" if pnl >= 0 else "red"

    lines = [
        f"Tickers:      {', '.join(state.config.tickers)}",
        f"Run Count:    {state.run_count}",
        f"Last Run:     {state.last_run.strftime('%Y-%m-%d %H:%M') if state.last_run else 'Never'}",
        f"Created:      {state.created_at.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Cash:         ${portfolio.cash:>12,.2f}",
        f"Holdings:     ${holdings_value:>12,.2f}",
        f"Total Value:  ${total_value:>12,.2f}",
        f"P&L:          [{color}]${pnl:>12,.2f} ({pnl_pct:+.2f}%)[/{color}]",
    ]

    console.print(Panel("\n".join(lines), title="Paper Portfolio", border_style="cyan"))


def _display_holdings(state: PaperTradingState) -> None:
    """Display current holdings table."""
    if not state.portfolio.positions:
        console.print("[dim]No holdings.[/dim]")
        return

    table = Table(title="Current Holdings", show_header=True, header_style="bold cyan")
    table.add_column("Ticker")
    table.add_column("Shares", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Market Value", justify="right")
    table.add_column("Unrealized P&L", justify="right")

    # Use latest snapshot for current prices if available
    latest_holdings = {}
    if state.snapshots:
        latest_holdings = state.snapshots[-1].holdings

    for ticker, pos in state.portfolio.positions.items():
        holding = latest_holdings.get(ticker)
        if holding:
            market_value = holding.market_value
            pnl = holding.unrealized_pnl
        else:
            market_value = pos.shares * pos.avg_cost
            pnl = 0.0

        color = "green" if pnl >= 0 else "red"
        table.add_row(
            ticker,
            str(pos.shares),
            f"${pos.avg_cost:,.2f}",
            f"${market_value:,.2f}",
            f"[{color}]${pnl:,.2f}[/{color}]",
        )

    console.print(table)


def _display_recent_trades(state: PaperTradingState, max_trades: int = 10) -> None:
    """Display recent trades."""
    if not state.trades:
        console.print("[dim]No trades yet.[/dim]")
        return

    trades = state.trades[-max_trades:]
    table = Table(
        title=f"Recent Trades (last {len(trades)} of {len(state.trades)})",
        show_header=True, header_style="bold cyan",
    )
    table.add_column("Date")
    table.add_column("Ticker")
    table.add_column("Action")
    table.add_column("Qty", justify="right")
    table.add_column("Price", justify="right")
    table.add_column("Value", justify="right")

    for trade in trades:
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


def _display_performance(state: PaperTradingState) -> None:
    """Display performance metrics if enough data."""
    if len(state.snapshots) < 2:
        return

    metrics = compute_metrics(state.snapshots, state.trades, state.config.initial_cash)

    table = Table(title="Performance Metrics", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    def _fmt(val, suffix="%"):
        if val is None:
            return "N/A"
        return f"{val:+.2f}{suffix}" if suffix == "%" else f"{val:.2f}"

    table.add_row("Total Return", _fmt(metrics.total_return_pct))
    table.add_row("Sharpe Ratio", _fmt(metrics.sharpe_ratio, ""))
    table.add_row("Max Drawdown", _fmt(metrics.max_drawdown_pct))
    table.add_row("Total Trades", str(metrics.total_trades))
    table.add_row("Win Rate", _fmt(metrics.win_rate_pct))
    table.add_row("Profit Factor", _fmt(metrics.profit_factor, ""))

    console.print(table)


# ── Commands ────────────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """Run one trading cycle."""
    tickers = [t.strip().upper() for t in args.ticker.split(",")]
    personas = [p.strip().lower() for p in args.personas.split(",")] if args.personas else None

    state = load_state(args.state_file)
    if state is None:
        config = PaperTradingConfig(
            tickers=tickers,
            initial_cash=args.cash,
            lookback_days=args.lookback,
            commission_rate=args.commission,
            slippage_rate=args.slippage,
            stop_loss_pct=args.stop_loss,
            trailing_stop_pct=args.trailing_stop,
            take_profit_pct=args.take_profit,
        )
        state = create_initial_state(config)
        console.print(f"[bold green]New paper portfolio created[/bold green] (${args.cash:,.0f})")
    else:
        console.print(f"[bold green]Loaded portfolio[/bold green] (run #{state.run_count + 1})")
        # Update tickers from CLI in case they changed
        state.config.tickers = tickers

    runner = PaperTradingRunner(
        state=state,
        model_name=args.model,
        model_provider=args.provider,
        use_llm=args.use_llm,
        personas=personas,
        show_reasoning=args.show_reasoning,
    )

    state = runner.run_cycle()
    save_state(state, args.state_file)

    console.print()
    _display_portfolio(state)
    console.print()
    _display_holdings(state)
    console.print()
    _display_recent_trades(state)
    console.print()
    _display_performance(state)
    console.print()
    console.print(f"[dim]State saved to {args.state_file}[/dim]")


def cmd_status(args: argparse.Namespace) -> None:
    """Show current portfolio without trading."""
    state = load_state(args.state_file)
    if state is None:
        console.print(f"[bold red]No portfolio found at {args.state_file}[/bold red]")
        console.print("Run 'python -m src.paper_trader run --ticker AAPL' to start.")
        sys.exit(1)

    console.print()
    _display_portfolio(state)
    console.print()
    _display_holdings(state)
    console.print()
    _display_recent_trades(state)
    console.print()
    _display_performance(state)
    console.print()


def cmd_reset(args: argparse.Namespace) -> None:
    """Reset portfolio to fresh state."""
    tickers = [t.strip().upper() for t in args.ticker.split(",")]
    config = PaperTradingConfig(
        tickers=tickers,
        initial_cash=args.cash,
        lookback_days=args.lookback,
        commission_rate=args.commission,
        slippage_rate=args.slippage,
        stop_loss_pct=args.stop_loss,
        trailing_stop_pct=args.trailing_stop,
        take_profit_pct=args.take_profit,
    )
    state = create_initial_state(config)
    save_state(state, args.state_file)
    console.print(f"[bold green]Portfolio reset[/bold green] — ${args.cash:,.0f} cash, tickers: {', '.join(tickers)}")
    console.print(f"[dim]State saved to {args.state_file}[/dim]")


# ── Main ────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    if hasattr(args, "debug") and args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    console.print("\n[bold green]AI Hedge Fund — Paper Trader[/bold green]")

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "reset":
        cmd_reset(args)
    else:
        console.print("Usage: python -m src.paper_trader {run|status|reset} [options]")
        console.print("  run    — Run one trading cycle")
        console.print("  status — Show current portfolio")
        console.print("  reset  — Reset portfolio to fresh state")
        sys.exit(1)


if __name__ == "__main__":
    main()
