"""Tests for BacktestEngine with mocked workflow."""
from datetime import date
from unittest.mock import patch

import pytest

from src.backtest.engine import BacktestEngine


def _mock_run_hedge_fund(tickers, start_date, end_date, portfolio, **kwargs):
    """Simulate a workflow that always buys 5 shares of each ticker."""
    prices = {"AAPL": 150.0, "MSFT": 400.0}
    positions = []
    for t in tickers:
        if t in prices:
            positions.append({
                "ticker": t, "action": "buy", "quantity": 5,
                "confidence": 80, "reasoning": "test",
            })
    return {
        "data": {
            "portfolio_output": {
                "positions": positions,
                "cash_remaining": portfolio.get("cash", 100_000),
                "total_value": portfolio.get("total_value", 100_000),
            },
            "current_prices": {t: prices.get(t, 100.0) for t in tickers},
        }
    }


class TestGenerateStepDates:
    def test_weekly(self):
        engine = BacktestEngine(
            tickers=["AAPL"], start_date="2024-01-01",
            end_date="2024-01-31", frequency="weekly",
        )
        dates = engine._generate_step_dates()
        assert len(dates) > 0
        # Weekly (W-FRI) should give Fridays
        for d in dates:
            assert d.weekday() == 4  # Friday

    def test_daily(self):
        engine = BacktestEngine(
            tickers=["AAPL"], start_date="2024-01-01",
            end_date="2024-01-05", frequency="daily",
        )
        dates = engine._generate_step_dates()
        assert len(dates) > 0
        for d in dates:
            assert d.weekday() < 5  # Business days only

    def test_monthly(self):
        engine = BacktestEngine(
            tickers=["AAPL"], start_date="2024-01-01",
            end_date="2024-06-30", frequency="monthly",
        )
        dates = engine._generate_step_dates()
        assert len(dates) >= 5  # Jan-Jun = at least 5 months

    def test_invalid_frequency(self):
        engine = BacktestEngine(
            tickers=["AAPL"], start_date="2024-01-01",
            end_date="2024-01-31", frequency="biweekly",
        )
        with pytest.raises(ValueError, match="Unknown frequency"):
            engine._generate_step_dates()

    def test_no_dates_in_range(self):
        engine = BacktestEngine(
            tickers=["AAPL"], start_date="2024-01-01",
            end_date="2024-01-01", frequency="weekly",
        )
        dates = engine._generate_step_dates()
        # A single day won't produce a Friday
        assert len(dates) == 0


class TestEngineRun:
    @patch("src.backtest.engine.run_hedge_fund", side_effect=_mock_run_hedge_fund)
    @patch("src.backtest.engine.compute_benchmark", return_value=None)
    def test_basic_run(self, mock_bench, mock_workflow):
        engine = BacktestEngine(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            initial_cash=100_000,
            frequency="weekly",
            benchmark_ticker=None,
        )
        result = engine.run()

        assert result.tickers == ["AAPL"]
        assert result.initial_cash == 100_000
        assert len(result.snapshots) > 0
        assert len(result.trades) > 0
        assert result.metrics is not None
        assert result.metrics.total_return_pct is not None

    @patch("src.backtest.engine.run_hedge_fund", side_effect=_mock_run_hedge_fund)
    @patch("src.backtest.engine.compute_benchmark", return_value=None)
    def test_trades_recorded(self, mock_bench, mock_workflow):
        engine = BacktestEngine(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            frequency="weekly",
            benchmark_ticker=None,
        )
        result = engine.run()

        # Each step buys 5 shares of AAPL
        for trade in result.trades:
            assert trade.ticker == "AAPL"
            assert trade.action == "buy"
            assert trade.price == 150.0

    @patch("src.backtest.engine.run_hedge_fund", side_effect=_mock_run_hedge_fund)
    @patch("src.backtest.engine.compute_benchmark", return_value=None)
    def test_multi_ticker(self, mock_bench, mock_workflow):
        engine = BacktestEngine(
            tickers=["AAPL", "MSFT"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            frequency="weekly",
            benchmark_ticker=None,
        )
        result = engine.run()

        traded_tickers = {t.ticker for t in result.trades}
        assert "AAPL" in traded_tickers
        assert "MSFT" in traded_tickers

    @patch("src.backtest.engine.run_hedge_fund", side_effect=Exception("API error"))
    @patch("src.backtest.engine.compute_benchmark", return_value=None)
    def test_handles_workflow_error(self, mock_bench, mock_workflow):
        """Engine should log error and continue, not crash."""
        engine = BacktestEngine(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            frequency="weekly",
            benchmark_ticker=None,
        )
        result = engine.run()

        # Should still produce snapshots (with unchanged state)
        assert len(result.snapshots) > 0
        assert result.trades == []
        assert result.final_value == 100_000

    @patch("src.backtest.engine.run_hedge_fund", side_effect=_mock_run_hedge_fund)
    @patch("src.backtest.engine.compute_benchmark", return_value=None)
    def test_cash_depletes_over_steps(self, mock_bench, mock_workflow):
        engine = BacktestEngine(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-02-29",
            initial_cash=5_000,
            frequency="weekly",
            benchmark_ticker=None,
        )
        result = engine.run()

        # With $5000 and buying 5 shares at $150/step, cash runs out
        final_snap = result.snapshots[-1]
        assert final_snap.cash < 5_000

    @patch("src.backtest.engine.run_hedge_fund", side_effect=_mock_run_hedge_fund)
    @patch("src.backtest.engine.compute_benchmark", return_value=None)
    def test_no_trading_dates_raises(self, mock_bench, mock_workflow):
        engine = BacktestEngine(
            tickers=["AAPL"],
            start_date="2024-01-01",
            end_date="2024-01-01",
            frequency="weekly",
            benchmark_ticker=None,
        )
        with pytest.raises(ValueError, match="No trading dates"):
            engine.run()
