"""Paper trading state model and persistence."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from src.backtest.models import PortfolioSnapshot, Trade
from src.backtest.portfolio_tracker import PortfolioTracker

logger = logging.getLogger(__name__)


class PositionState(BaseModel):
    """Persisted position for a single ticker."""
    shares: int
    avg_cost: float


class PortfolioState(BaseModel):
    """Persisted portfolio — cash + positions."""
    cash: float
    positions: dict[str, PositionState] = Field(default_factory=dict)


class PaperTradingConfig(BaseModel):
    """Configuration for a paper trading session."""
    tickers: list[str]
    initial_cash: float = 100_000
    lookback_days: int = 90
    commission_rate: float = 0.001
    slippage_rate: float = 0.00005


class PaperTradingState(BaseModel):
    """Full paper trading state — serialized to/from JSON."""
    version: int = 1
    created_at: datetime
    last_run: Optional[datetime] = None
    run_count: int = 0
    config: PaperTradingConfig
    portfolio: PortfolioState
    trades: list[Trade] = Field(default_factory=list)
    snapshots: list[PortfolioSnapshot] = Field(default_factory=list)


def create_initial_state(config: PaperTradingConfig) -> PaperTradingState:
    """Create a fresh paper trading state."""
    return PaperTradingState(
        created_at=datetime.now(),
        config=config,
        portfolio=PortfolioState(cash=config.initial_cash),
    )


def load_state(path: str) -> Optional[PaperTradingState]:
    """Load paper trading state from a JSON file. Returns None if file doesn't exist."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        return PaperTradingState.model_validate(data)
    except Exception as e:
        logger.error(f"Failed to load state from {path}: {e}")
        return None


def save_state(state: PaperTradingState, path: str) -> None:
    """Save paper trading state to a JSON file."""
    data = state.model_dump(mode="json")
    Path(path).write_text(json.dumps(data, indent=2))


def to_tracker(state: PaperTradingState) -> PortfolioTracker:
    """Reconstruct a PortfolioTracker from saved state."""
    tracker = PortfolioTracker(
        initial_cash=state.config.initial_cash,
        commission_rate=state.config.commission_rate,
        slippage_rate=state.config.slippage_rate,
    )
    tracker.cash = state.portfolio.cash
    tracker.positions = {
        ticker: {"shares": pos.shares, "avg_cost": pos.avg_cost}
        for ticker, pos in state.portfolio.positions.items()
    }
    tracker.trades = list(state.trades)
    tracker.snapshots = list(state.snapshots)
    return tracker


def from_tracker(tracker: PortfolioTracker, state: PaperTradingState) -> PaperTradingState:
    """Update state from tracker after a trading cycle."""
    state.portfolio.cash = tracker.cash
    state.portfolio.positions = {
        ticker: PositionState(shares=pos["shares"], avg_cost=pos["avg_cost"])
        for ticker, pos in tracker.positions.items()
    }
    state.trades = tracker.trades
    state.snapshots = tracker.snapshots
    state.run_count += 1
    state.last_run = datetime.now()
    return state
