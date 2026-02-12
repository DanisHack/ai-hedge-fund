"""Backtest-specific data classes."""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Trade(BaseModel):
    """A single executed trade."""
    date: date
    ticker: str
    action: str  # "buy" or "sell"
    quantity: int
    price: float
    total_value: float
    commission: float = 0.0
    slippage: float = 0.0


class HoldingDetail(BaseModel):
    """Snapshot of a single holding at a point in time."""
    shares: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float


class PortfolioSnapshot(BaseModel):
    """Portfolio state at a single point in time."""
    date: date
    cash: float
    holdings: dict[str, HoldingDetail] = Field(default_factory=dict)
    total_value: float
    daily_return: Optional[float] = None


class PerformanceMetrics(BaseModel):
    """Aggregate performance statistics for a backtest."""
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: float
    max_drawdown_start: Optional[date] = None
    max_drawdown_end: Optional[date] = None
    win_rate_pct: Optional[float] = None
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win_pct: Optional[float] = None
    avg_loss_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    volatility_annual_pct: Optional[float] = None
    calmar_ratio: Optional[float] = None


class BenchmarkResult(BaseModel):
    """Buy-and-hold benchmark performance."""
    ticker: str
    start_price: float
    end_price: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: Optional[float] = None
    max_drawdown_pct: float


class BacktestResult(BaseModel):
    """Complete backtest output."""
    tickers: list[str]
    start_date: date
    end_date: date
    frequency: str
    initial_cash: float
    final_value: float
    snapshots: list[PortfolioSnapshot] = Field(default_factory=list)
    trades: list[Trade] = Field(default_factory=list)
    metrics: PerformanceMetrics
    benchmark: Optional[BenchmarkResult] = None
