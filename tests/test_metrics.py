"""Tests for compute_metrics and trade analysis."""
from datetime import date

import pytest

from src.backtest.metrics import compute_metrics, _analyze_trades
from src.backtest.models import PerformanceMetrics, PortfolioSnapshot, Trade


def _make_snapshot(d: date, total_value: float, daily_return=None) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        date=d, cash=total_value, holdings={},
        total_value=total_value, daily_return=daily_return,
    )


class TestComputeMetrics:
    def test_empty_snapshots(self):
        m = compute_metrics([], [], 100_000)
        assert m.total_return_pct == 0.0
        assert m.total_trades == 0

    def test_total_return(self):
        snapshots = [
            _make_snapshot(date(2024, 1, 1), 100_000),
            _make_snapshot(date(2024, 7, 1), 110_000, daily_return=0.1),
        ]
        m = compute_metrics(snapshots, [], 100_000)
        assert m.total_return_pct == 10.0

    def test_negative_return(self):
        snapshots = [
            _make_snapshot(date(2024, 1, 1), 100_000),
            _make_snapshot(date(2024, 7, 1), 90_000, daily_return=-0.1),
        ]
        m = compute_metrics(snapshots, [], 100_000)
        assert m.total_return_pct == -10.0

    def test_max_drawdown(self):
        # Goes up to 120k, drops to 90k, recovers to 100k
        snapshots = [
            _make_snapshot(date(2024, 1, 1), 100_000),
            _make_snapshot(date(2024, 2, 1), 120_000, daily_return=0.2),
            _make_snapshot(date(2024, 3, 1), 90_000, daily_return=-0.25),
            _make_snapshot(date(2024, 4, 1), 100_000, daily_return=0.111),
        ]
        m = compute_metrics(snapshots, [], 100_000)
        # Drawdown: (90000-120000)/120000 = -25%
        assert m.max_drawdown_pct == -25.0
        assert m.max_drawdown_start == date(2024, 2, 1)
        assert m.max_drawdown_end == date(2024, 3, 1)

    def test_no_drawdown(self):
        snapshots = [
            _make_snapshot(date(2024, 1, 1), 100_000),
            _make_snapshot(date(2024, 2, 1), 110_000, daily_return=0.1),
            _make_snapshot(date(2024, 3, 1), 120_000, daily_return=0.0909),
        ]
        m = compute_metrics(snapshots, [], 100_000)
        assert m.max_drawdown_pct == 0.0

    def test_volatility_computed(self):
        snapshots = [
            _make_snapshot(date(2024, 1, 1), 100_000),
            _make_snapshot(date(2024, 1, 2), 101_000, daily_return=0.01),
            _make_snapshot(date(2024, 1, 3), 99_000, daily_return=-0.0198),
            _make_snapshot(date(2024, 1, 4), 102_000, daily_return=0.0303),
        ]
        m = compute_metrics(snapshots, [], 100_000)
        assert m.volatility_annual_pct is not None
        assert m.volatility_annual_pct > 0

    def test_sharpe_ratio_computed(self):
        snapshots = [
            _make_snapshot(date(2024, 1, 1), 100_000),
            _make_snapshot(date(2024, 1, 2), 101_000, daily_return=0.01),
            _make_snapshot(date(2024, 1, 3), 102_000, daily_return=0.0099),
            _make_snapshot(date(2024, 7, 1), 120_000, daily_return=0.01),
        ]
        m = compute_metrics(snapshots, [], 100_000)
        assert m.sharpe_ratio is not None

    def test_calmar_ratio(self):
        snapshots = [
            _make_snapshot(date(2024, 1, 1), 100_000),
            _make_snapshot(date(2024, 6, 1), 120_000, daily_return=0.2),
            _make_snapshot(date(2024, 7, 1), 100_000, daily_return=-0.167),
            _make_snapshot(date(2024, 12, 31), 130_000, daily_return=0.3),
        ]
        m = compute_metrics(snapshots, [], 100_000)
        assert m.calmar_ratio is not None
        assert m.calmar_ratio > 0


class TestAnalyzeTrades:
    def test_no_trades(self):
        win_rate, winning, losing, avg_win, avg_loss, pf = _analyze_trades([])
        assert win_rate is None
        assert winning == 0
        assert losing == 0

    def test_buys_only_no_round_trips(self):
        trades = [
            Trade(date=date(2024, 1, 1), ticker="AAPL", action="buy",
                  quantity=10, price=100.0, total_value=1000.0),
        ]
        win_rate, winning, losing, _, _, _ = _analyze_trades(trades)
        assert win_rate is None
        assert winning == 0

    def test_winning_trade(self):
        trades = [
            Trade(date=date(2024, 1, 1), ticker="AAPL", action="buy",
                  quantity=10, price=100.0, total_value=1000.0),
            Trade(date=date(2024, 2, 1), ticker="AAPL", action="sell",
                  quantity=10, price=120.0, total_value=1200.0),
        ]
        win_rate, winning, losing, avg_win, avg_loss, pf = _analyze_trades(trades)
        assert winning == 1
        assert losing == 0
        assert win_rate == 100.0
        assert avg_win == pytest.approx(20.0)  # (120-100)/100 * 100

    def test_losing_trade(self):
        trades = [
            Trade(date=date(2024, 1, 1), ticker="AAPL", action="buy",
                  quantity=10, price=100.0, total_value=1000.0),
            Trade(date=date(2024, 2, 1), ticker="AAPL", action="sell",
                  quantity=10, price=80.0, total_value=800.0),
        ]
        win_rate, winning, losing, avg_win, avg_loss, pf = _analyze_trades(trades)
        assert winning == 0
        assert losing == 1
        assert win_rate == 0.0
        assert avg_loss == pytest.approx(-20.0)

    def test_mixed_win_loss(self):
        trades = [
            Trade(date=date(2024, 1, 1), ticker="AAPL", action="buy",
                  quantity=10, price=100.0, total_value=1000.0),
            Trade(date=date(2024, 2, 1), ticker="AAPL", action="sell",
                  quantity=10, price=120.0, total_value=1200.0),
            Trade(date=date(2024, 3, 1), ticker="MSFT", action="buy",
                  quantity=5, price=300.0, total_value=1500.0),
            Trade(date=date(2024, 4, 1), ticker="MSFT", action="sell",
                  quantity=5, price=250.0, total_value=1250.0),
        ]
        win_rate, winning, losing, avg_win, avg_loss, pf = _analyze_trades(trades)
        assert winning == 1
        assert losing == 1
        assert win_rate == 50.0
        # profit_factor = total_gains / total_losses = 20% / 16.67%
        assert pf is not None
        assert pf > 1.0

    def test_fifo_ordering(self):
        """First buy is matched to first sell (FIFO)."""
        trades = [
            Trade(date=date(2024, 1, 1), ticker="AAPL", action="buy",
                  quantity=10, price=100.0, total_value=1000.0),
            Trade(date=date(2024, 2, 1), ticker="AAPL", action="buy",
                  quantity=10, price=200.0, total_value=2000.0),
            Trade(date=date(2024, 3, 1), ticker="AAPL", action="sell",
                  quantity=10, price=150.0, total_value=1500.0),
        ]
        win_rate, winning, losing, avg_win, avg_loss, _ = _analyze_trades(trades)
        # First buy at 100, sold at 150 = +50% win
        assert winning == 1
        assert losing == 0
        assert avg_win == pytest.approx(50.0)

    def test_partial_sell_matches_fifo(self):
        """Sell of 5 from a buy of 10 still counts as a round trip."""
        trades = [
            Trade(date=date(2024, 1, 1), ticker="AAPL", action="buy",
                  quantity=10, price=100.0, total_value=1000.0),
            Trade(date=date(2024, 2, 1), ticker="AAPL", action="sell",
                  quantity=5, price=110.0, total_value=550.0),
        ]
        win_rate, winning, losing, _, _, _ = _analyze_trades(trades)
        assert winning == 1
        assert losing == 0
