"""Core backtest loop engine."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from src.backtest.metrics import compute_benchmark, compute_metrics
from src.backtest.models import BacktestResult
from src.backtest.portfolio_tracker import PortfolioTracker
from src.graph.workflow import run_hedge_fund

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Step through historical dates, invoke the hedge fund workflow, and track portfolio."""

    def __init__(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        initial_cash: float = 100_000,
        frequency: str = "weekly",
        lookback_days: int = 90,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        show_reasoning: bool = False,
        use_llm: bool = False,
        personas: list[str] | None = None,
        benchmark_ticker: Optional[str] = "SPY",
        commission_rate: float = 0.001,
        slippage_rate: float = 0.00005,
    ) -> None:
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.frequency = frequency
        self.lookback_days = lookback_days
        self.model_name = model_name
        self.model_provider = model_provider
        self.show_reasoning = show_reasoning
        self.use_llm = use_llm
        self.personas = personas
        self.benchmark_ticker = benchmark_ticker
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate

    def _generate_step_dates(self) -> list[date]:
        """Generate trading dates based on frequency."""
        freq_map = {
            "daily": "B",       # every business day
            "weekly": "W-FRI",  # every Friday
            "monthly": "BME",   # last business day of month
        }
        freq = freq_map.get(self.frequency)
        if freq is None:
            raise ValueError(f"Unknown frequency '{self.frequency}'. Use: daily, weekly, monthly")

        dates = pd.bdate_range(start=self.start_date, end=self.end_date, freq=freq)
        return [d.date() for d in dates]

    def run(self) -> BacktestResult:
        """Execute the backtest loop."""
        step_dates = self._generate_step_dates()
        if not step_dates:
            raise ValueError(f"No trading dates between {self.start_date} and {self.end_date}")

        tracker = PortfolioTracker(self.initial_cash, commission_rate=self.commission_rate, slippage_rate=self.slippage_rate)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task(
                f"Backtesting {', '.join(self.tickers)}", total=len(step_dates)
            )

            for step_date in step_dates:
                lookback_start = step_date - timedelta(days=self.lookback_days)
                portfolio_dict = tracker.get_portfolio_dict()

                try:
                    progress.update(task, description=f"Step {step_date}")

                    result = run_hedge_fund(
                        tickers=self.tickers,
                        start_date=lookback_start.strftime("%Y-%m-%d"),
                        end_date=step_date.strftime("%Y-%m-%d"),
                        portfolio=portfolio_dict,
                        model_name=self.model_name,
                        model_provider=self.model_provider,
                        show_reasoning=self.show_reasoning,
                        use_llm=self.use_llm,
                        personas=self.personas,
                    )

                    data = result.get("data", {})
                    portfolio_output = data.get("portfolio_output", {})
                    current_prices = data.get("current_prices", {})

                    tracker.apply_trades(portfolio_output, current_prices, step_date)
                    tracker.take_snapshot(step_date, current_prices)

                except Exception as e:
                    logger.error(f"Error on {step_date}: {e}")
                    # Snapshot with unchanged state
                    tracker.take_snapshot(step_date, {})

                progress.advance(task)

        # Compute metrics
        metrics = compute_metrics(tracker.snapshots, tracker.trades, self.initial_cash)

        # Benchmark
        benchmark = None
        if self.benchmark_ticker:
            benchmark = compute_benchmark(
                self.benchmark_ticker,
                self.start_date,
                self.end_date,
                self.initial_cash,
            )

        final_value = tracker.snapshots[-1].total_value if tracker.snapshots else self.initial_cash

        return BacktestResult(
            tickers=self.tickers,
            start_date=date.fromisoformat(self.start_date),
            end_date=date.fromisoformat(self.end_date),
            frequency=self.frequency,
            initial_cash=self.initial_cash,
            final_value=final_value,
            snapshots=tracker.snapshots,
            trades=tracker.trades,
            metrics=metrics,
            benchmark=benchmark,
        )
