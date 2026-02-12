"""Tests for PortfolioTracker."""
from datetime import date

import pytest

from src.backtest.portfolio_tracker import PortfolioTracker


class TestInit:
    def test_initial_state(self):
        tracker = PortfolioTracker(100_000)
        assert tracker.cash == 100_000
        assert tracker.initial_cash == 100_000
        assert tracker.positions == {}
        assert tracker.trades == []
        assert tracker.snapshots == []


class TestGetPortfolioDict:
    def test_empty_portfolio(self):
        tracker = PortfolioTracker(50_000)
        d = tracker.get_portfolio_dict()
        assert d == {"cash": 50_000, "positions": {}, "total_value": 50_000}

    def test_with_positions(self):
        tracker = PortfolioTracker(50_000)
        tracker.positions["AAPL"] = {"shares": 10, "avg_cost": 150.0}
        d = tracker.get_portfolio_dict()
        assert d["cash"] == 50_000
        assert d["positions"]["AAPL"] == {"shares": 10, "avg_cost": 150.0}
        assert d["total_value"] == 50_000 + 10 * 150.0


class TestApplyTrades:
    def test_buy_new_position(self):
        tracker = PortfolioTracker(100_000)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        assert tracker.cash == 100_000 - 10 * 150.0
        assert tracker.positions["AAPL"]["shares"] == 10
        assert tracker.positions["AAPL"]["avg_cost"] == 150.0
        assert len(tracker.trades) == 1
        assert tracker.trades[0].action == "buy"
        assert tracker.trades[0].total_value == 1500.0

    def test_buy_adds_to_existing_position(self):
        tracker = PortfolioTracker(100_000)
        tracker.positions["AAPL"] = {"shares": 10, "avg_cost": 100.0}

        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 200.0}, date(2024, 1, 1))

        assert tracker.positions["AAPL"]["shares"] == 20
        # Weighted avg: (10*100 + 10*200) / 20 = 150
        assert tracker.positions["AAPL"]["avg_cost"] == 150.0

    def test_buy_insufficient_cash_buys_partial(self):
        tracker = PortfolioTracker(500)  # Only $500 cash
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        # Can only afford 3 shares at $150 = $450
        assert tracker.positions["AAPL"]["shares"] == 3
        assert tracker.cash == 500 - 3 * 150.0
        assert tracker.trades[0].quantity == 3

    def test_buy_zero_affordable_skipped(self):
        tracker = PortfolioTracker(10)  # Only $10
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 5},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        assert "AAPL" not in tracker.positions
        assert tracker.trades == []
        assert tracker.cash == 10

    def test_sell_full_position(self):
        tracker = PortfolioTracker(50_000)
        tracker.positions["AAPL"] = {"shares": 10, "avg_cost": 100.0}

        output = {
            "positions": [
                {"ticker": "AAPL", "action": "sell", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 200.0}, date(2024, 1, 1))

        assert "AAPL" not in tracker.positions  # Fully sold
        assert tracker.cash == 50_000 + 10 * 200.0
        assert len(tracker.trades) == 1
        assert tracker.trades[0].action == "sell"
        assert tracker.trades[0].total_value == 2000.0

    def test_sell_partial_position(self):
        tracker = PortfolioTracker(50_000)
        tracker.positions["AAPL"] = {"shares": 10, "avg_cost": 100.0}

        output = {
            "positions": [
                {"ticker": "AAPL", "action": "sell", "quantity": 5},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 200.0}, date(2024, 1, 1))

        assert tracker.positions["AAPL"]["shares"] == 5
        assert tracker.cash == 50_000 + 5 * 200.0

    def test_sell_more_than_held_caps_to_held(self):
        tracker = PortfolioTracker(50_000)
        tracker.positions["AAPL"] = {"shares": 5, "avg_cost": 100.0}

        output = {
            "positions": [
                {"ticker": "AAPL", "action": "sell", "quantity": 100},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 200.0}, date(2024, 1, 1))

        assert "AAPL" not in tracker.positions
        assert tracker.cash == 50_000 + 5 * 200.0
        assert tracker.trades[0].quantity == 5

    def test_sell_no_position_is_noop(self):
        tracker = PortfolioTracker(50_000)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "sell", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 200.0}, date(2024, 1, 1))

        assert tracker.trades == []
        assert tracker.cash == 50_000

    def test_hold_is_noop(self):
        tracker = PortfolioTracker(100_000)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "hold", "quantity": 0},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        assert tracker.trades == []
        assert tracker.cash == 100_000

    def test_no_price_skipped(self):
        tracker = PortfolioTracker(100_000)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {}, date(2024, 1, 1))

        assert tracker.trades == []

    def test_zero_quantity_skipped(self):
        tracker = PortfolioTracker(100_000)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 0},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        assert tracker.trades == []

    def test_multiple_tickers(self):
        tracker = PortfolioTracker(100_000)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 5},
                {"ticker": "MSFT", "action": "buy", "quantity": 3},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0, "MSFT": 400.0}, date(2024, 1, 1))

        assert tracker.positions["AAPL"]["shares"] == 5
        assert tracker.positions["MSFT"]["shares"] == 3
        assert len(tracker.trades) == 2
        assert tracker.cash == 100_000 - (5 * 150.0 + 3 * 400.0)


class TestTransactionCosts:
    def test_buy_with_commission(self):
        tracker = PortfolioTracker(100_000, commission_rate=0.001, slippage_rate=0.00005)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        gross_cost = 10 * 150.0
        commission = gross_cost * 0.001
        slippage = gross_cost * 0.00005
        expected_cash = 100_000 - gross_cost - commission - slippage
        assert tracker.cash == pytest.approx(expected_cash)
        assert tracker.positions["AAPL"]["shares"] == 10

    def test_sell_with_costs(self):
        tracker = PortfolioTracker(50_000, commission_rate=0.001, slippage_rate=0.00005)
        tracker.positions["AAPL"] = {"shares": 10, "avg_cost": 100.0}

        output = {
            "positions": [
                {"ticker": "AAPL", "action": "sell", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 200.0}, date(2024, 1, 1))

        gross_proceeds = 10 * 200.0
        commission = gross_proceeds * 0.001
        slippage = gross_proceeds * 0.00005
        expected_cash = 50_000 + gross_proceeds - commission - slippage
        assert tracker.cash == pytest.approx(expected_cash)

    def test_trade_records_costs(self):
        tracker = PortfolioTracker(100_000, commission_rate=0.001, slippage_rate=0.00005)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        trade = tracker.trades[0]
        assert trade.commission == pytest.approx(1500.0 * 0.001)
        assert trade.slippage == pytest.approx(1500.0 * 0.00005)

    def test_buy_partial_with_costs(self):
        # With costs, $500 buys fewer shares at $150/share
        tracker = PortfolioTracker(500, commission_rate=0.01, slippage_rate=0.005)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        # effective price = 150 * (1 + 0.01 + 0.005) = 150 * 1.015 = 152.25
        # affordable = int(500 / 152.25) = 3
        assert tracker.positions["AAPL"]["shares"] == 3
        gross_cost = 3 * 150.0
        commission = gross_cost * 0.01
        slippage = gross_cost * 0.005
        assert tracker.cash == pytest.approx(500 - gross_cost - commission - slippage)

    def test_zero_costs_matches_original(self):
        tracker = PortfolioTracker(100_000, commission_rate=0.0, slippage_rate=0.0)
        output = {
            "positions": [
                {"ticker": "AAPL", "action": "buy", "quantity": 10},
            ]
        }
        tracker.apply_trades(output, {"AAPL": 150.0}, date(2024, 1, 1))

        assert tracker.cash == 100_000 - 10 * 150.0
        assert tracker.positions["AAPL"]["shares"] == 10
        assert tracker.trades[0].commission == 0.0
        assert tracker.trades[0].slippage == 0.0


class TestTakeSnapshot:
    def test_first_snapshot_no_daily_return(self):
        tracker = PortfolioTracker(100_000)
        snap = tracker.take_snapshot(date(2024, 1, 1), {})
        assert snap.total_value == 100_000
        assert snap.daily_return is None
        assert snap.holdings == {}

    def test_snapshot_with_holdings(self):
        tracker = PortfolioTracker(50_000)
        tracker.positions["AAPL"] = {"shares": 10, "avg_cost": 100.0}

        snap = tracker.take_snapshot(date(2024, 1, 1), {"AAPL": 150.0})
        assert snap.total_value == 50_000 + 10 * 150.0
        assert "AAPL" in snap.holdings
        assert snap.holdings["AAPL"].shares == 10
        assert snap.holdings["AAPL"].current_price == 150.0
        assert snap.holdings["AAPL"].market_value == 1500.0
        assert snap.holdings["AAPL"].unrealized_pnl == 500.0  # (150-100)*10

    def test_daily_return_computed_from_previous(self):
        tracker = PortfolioTracker(100_000)
        tracker.take_snapshot(date(2024, 1, 1), {})

        # Simulate a gain
        tracker.cash = 110_000
        snap = tracker.take_snapshot(date(2024, 1, 2), {})
        assert snap.daily_return == pytest.approx(0.10)  # 10% gain

    def test_missing_price_falls_back_to_avg_cost(self):
        tracker = PortfolioTracker(50_000)
        tracker.positions["AAPL"] = {"shares": 10, "avg_cost": 100.0}

        snap = tracker.take_snapshot(date(2024, 1, 1), {})
        # Falls back to avg_cost=100, so market_value = 10*100
        assert snap.holdings["AAPL"].current_price == 100.0
        assert snap.holdings["AAPL"].unrealized_pnl == 0.0

    def test_snapshots_accumulate(self):
        tracker = PortfolioTracker(100_000)
        tracker.take_snapshot(date(2024, 1, 1), {})
        tracker.take_snapshot(date(2024, 1, 2), {})
        tracker.take_snapshot(date(2024, 1, 3), {})
        assert len(tracker.snapshots) == 3
