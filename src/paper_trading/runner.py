"""Paper trading runner — executes a single trading cycle."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

from src.graph.workflow import run_hedge_fund
from src.paper_trading.state import PaperTradingState, from_tracker, to_tracker

logger = logging.getLogger(__name__)


class PaperTradingRunner:
    """Run a single paper trading cycle against live market data."""

    def __init__(
        self,
        state: PaperTradingState,
        model_name: str = "gpt-4o-mini",
        model_provider: str = "openai",
        use_llm: bool = False,
        personas: Optional[list[str]] = None,
        show_reasoning: bool = False,
    ) -> None:
        self.state = state
        self.model_name = model_name
        self.model_provider = model_provider
        self.use_llm = use_llm
        self.personas = personas
        self.show_reasoning = show_reasoning

    def run_cycle(self) -> PaperTradingState:
        """Execute one trading cycle: analyze → trade → snapshot → update state."""
        today = date.today()
        lookback_start = today - timedelta(days=self.state.config.lookback_days)

        tracker = to_tracker(self.state)
        portfolio_dict = tracker.get_portfolio_dict()

        result = run_hedge_fund(
            tickers=self.state.config.tickers,
            start_date=lookback_start.strftime("%Y-%m-%d"),
            end_date=today.strftime("%Y-%m-%d"),
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

        tracker.update_high_water_marks(current_prices)
        tracker.check_stop_orders(current_prices, today)
        tracker.apply_trades(portfolio_output, current_prices, today)
        tracker.take_snapshot(today, current_prices)

        return from_tracker(tracker, self.state)
