"""AI Hedge Fund — Main entry point."""
from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table

from src.config.settings import DEFAULT_MODEL_NAME, DEFAULT_MODEL_PROVIDER
from src.graph.workflow import run_hedge_fund

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Hedge Fund — Multi-agent stock analysis")
    parser.add_argument(
        "--ticker", "-t", type=str, required=True,
        help="Comma-separated stock tickers (e.g. AAPL,MSFT,NVDA)",
    )
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cash", type=float, default=100000, help="Starting cash (default: 100000)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="LLM model name")
    parser.add_argument("--provider", type=str, default=DEFAULT_MODEL_PROVIDER, help="LLM provider")
    parser.add_argument("--show-reasoning", action="store_true", default=True)
    parser.add_argument("--use-llm", action="store_true", default=False,
                        help="Use LLM reasoning for analyst agents (requires API key)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def display_results(final_state: dict) -> None:
    """Display portfolio results using rich tables."""
    data = final_state.get("data", {})

    # ── Analyst Signals ─────────────────────────────────────────
    console.print("\n[bold cyan]Analyst Signals[/bold cyan]")
    sig_table = Table(show_header=True, header_style="bold")
    sig_table.add_column("Agent")
    sig_table.add_column("Ticker")
    sig_table.add_column("Signal")
    sig_table.add_column("Confidence")
    sig_table.add_column("Reasoning", max_width=60)

    for agent_id, signals in data.get("analyst_signals", {}).items():
        for sig in signals:
            color = {"bullish": "green", "bearish": "red"}.get(sig.get("signal", ""), "yellow")
            sig_table.add_row(
                agent_id,
                sig.get("ticker", ""),
                f"[{color}]{sig.get('signal', '')}[/{color}]",
                f"{sig.get('confidence', 0)}%",
                str(sig.get("reasoning", ""))[:60],
            )
    console.print(sig_table)

    # ── Risk Adjusted ───────────────────────────────────────────
    risk_signals = data.get("risk_adjusted_signals", [])
    if risk_signals:
        console.print("\n[bold cyan]Risk-Adjusted Signals[/bold cyan]")
        risk_table = Table(show_header=True, header_style="bold")
        risk_table.add_column("Ticker")
        risk_table.add_column("Consensus")
        risk_table.add_column("Adj. Confidence")
        risk_table.add_column("Max Position")

        for rs in risk_signals:
            color = {"bullish": "green", "bearish": "red"}.get(rs.get("signal", ""), "yellow")
            risk_table.add_row(
                rs.get("ticker", ""),
                f"[{color}]{rs.get('signal', '')}[/{color}]",
                f"{rs.get('confidence', 0)}%",
                f"${rs.get('max_position_size', 0):,.0f}",
            )
        console.print(risk_table)

    # ── Portfolio Decisions ─────────────────────────────────────
    portfolio = data.get("portfolio_output", {})
    console.print("\n[bold cyan]Portfolio Decisions[/bold cyan]")
    port_table = Table(show_header=True, header_style="bold")
    port_table.add_column("Ticker")
    port_table.add_column("Action")
    port_table.add_column("Quantity")
    port_table.add_column("Confidence")
    port_table.add_column("Reasoning", max_width=60)

    for pos in portfolio.get("positions", []):
        color = {"buy": "green", "sell": "red"}.get(pos.get("action", ""), "yellow")
        port_table.add_row(
            pos.get("ticker", ""),
            f"[{color}]{pos.get('action', '')}[/{color}]",
            str(pos.get("quantity", 0)),
            f"{pos.get('confidence', 0)}%",
            str(pos.get("reasoning", ""))[:60],
        )
    console.print(port_table)

    console.print(f"\n[bold]Cash Remaining:[/bold] ${portfolio.get('cash_remaining', 0):,.2f}")
    console.print(f"[bold]Total Value:[/bold] ${portfolio.get('total_value', 0):,.2f}\n")


def main():
    args = parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    tickers = [t.strip().upper() for t in args.ticker.split(",")]
    portfolio = {"cash": args.cash, "positions": {}, "total_value": args.cash}

    console.print(f"\n[bold green]AI Hedge Fund[/bold green] — Analyzing {', '.join(tickers)}\n")

    try:
        final_state = run_hedge_fund(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            portfolio=portfolio,
            model_name=args.model,
            model_provider=args.provider,
            show_reasoning=args.show_reasoning,
            use_llm=args.use_llm,
        )
        display_results(final_state)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
