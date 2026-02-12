"""Tests for src/paper_trading — state persistence, runner, and CLI."""
from __future__ import annotations

import json
from datetime import date, datetime
from unittest.mock import patch


from src.backtest.models import HoldingDetail, PortfolioSnapshot, Trade
from src.paper_trading.state import (
    PaperTradingConfig,
    PaperTradingState,
    PortfolioState,
    PositionState,
    create_initial_state,
    from_tracker,
    load_state,
    save_state,
    to_tracker,
)


def _make_config(**overrides):
    defaults = {
        "tickers": ["AAPL", "MSFT"],
        "initial_cash": 100_000,
        "lookback_days": 90,
        "commission_rate": 0.001,
        "slippage_rate": 0.00005,
    }
    defaults.update(overrides)
    return PaperTradingConfig(**defaults)


def _make_state(**overrides):
    config = overrides.pop("config", _make_config())
    defaults = {
        "created_at": datetime(2024, 1, 1, 10, 0),
        "config": config,
        "portfolio": PortfolioState(cash=config.initial_cash),
    }
    defaults.update(overrides)
    return PaperTradingState(**defaults)


class TestStateModels:
    def test_create_initial_state(self):
        config = _make_config()
        state = create_initial_state(config)

        assert state.portfolio.cash == 100_000
        assert state.portfolio.positions == {}
        assert state.run_count == 0
        assert state.last_run is None
        assert state.trades == []
        assert state.snapshots == []
        assert state.config.tickers == ["AAPL", "MSFT"]

    def test_create_initial_state_custom_cash(self):
        config = _make_config(initial_cash=50_000)
        state = create_initial_state(config)
        assert state.portfolio.cash == 50_000


class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        state = _make_state()
        state.portfolio.cash = 91_000
        state.portfolio.positions = {
            "AAPL": PositionState(shares=50, avg_cost=180.0),
        }
        state.trades = [
            Trade(date=date(2024, 1, 15), ticker="AAPL", action="buy",
                  quantity=50, price=180.0, total_value=9000.0),
        ]
        state.snapshots = [
            PortfolioSnapshot(
                date=date(2024, 1, 15), cash=91_000,
                holdings={"AAPL": HoldingDetail(shares=50, avg_cost=180.0,
                                                 current_price=182.0, market_value=9100.0,
                                                 unrealized_pnl=100.0)},
                total_value=100_100,
            ),
        ]
        state.run_count = 1
        state.last_run = datetime(2024, 1, 15, 10, 30)

        path = str(tmp_path / "test_state.json")
        save_state(state, path)
        loaded = load_state(path)

        assert loaded is not None
        assert loaded.portfolio.cash == 91_000
        assert loaded.portfolio.positions["AAPL"].shares == 50
        assert loaded.portfolio.positions["AAPL"].avg_cost == 180.0
        assert loaded.run_count == 1
        assert len(loaded.trades) == 1
        assert loaded.trades[0].ticker == "AAPL"
        assert len(loaded.snapshots) == 1
        assert loaded.snapshots[0].total_value == 100_100

    def test_load_nonexistent_returns_none(self, tmp_path):
        result = load_state(str(tmp_path / "does_not_exist.json"))
        assert result is None

    def test_load_corrupted_returns_none(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not valid json{{{")
        result = load_state(path)
        assert result is None

    def test_save_creates_file(self, tmp_path):
        state = _make_state()
        path = str(tmp_path / "new_state.json")
        save_state(state, path)

        with open(path) as f:
            data = json.load(f)
        assert data["version"] == 1
        assert data["config"]["tickers"] == ["AAPL", "MSFT"]


class TestTrackerConversion:
    def test_to_tracker(self):
        state = _make_state()
        state.portfolio.cash = 80_000
        state.portfolio.positions = {
            "AAPL": PositionState(shares=100, avg_cost=200.0),
        }

        tracker = to_tracker(state)

        assert tracker.cash == 80_000
        assert tracker.positions["AAPL"]["shares"] == 100
        assert tracker.positions["AAPL"]["avg_cost"] == 200.0
        assert tracker.commission_rate == 0.001

    def test_from_tracker(self):
        state = _make_state()
        tracker = to_tracker(state)

        # Simulate trades
        tracker.cash = 85_000
        tracker.positions = {"MSFT": {"shares": 20, "avg_cost": 350.0}}
        tracker.trades.append(
            Trade(date=date(2024, 2, 1), ticker="MSFT", action="buy",
                  quantity=20, price=350.0, total_value=7000.0),
        )
        tracker.snapshots.append(
            PortfolioSnapshot(date=date(2024, 2, 1), cash=85_000,
                              total_value=92_000),
        )

        updated = from_tracker(tracker, state)

        assert updated.portfolio.cash == 85_000
        assert updated.portfolio.positions["MSFT"].shares == 20
        assert updated.run_count == 1
        assert updated.last_run is not None
        assert len(updated.trades) == 1
        assert len(updated.snapshots) == 1

    def test_roundtrip_tracker_preserves_state(self):
        state = _make_state()
        state.portfolio.cash = 75_000
        state.portfolio.positions = {
            "AAPL": PositionState(shares=50, avg_cost=180.0),
            "MSFT": PositionState(shares=30, avg_cost=350.0),
        }

        tracker = to_tracker(state)
        updated = from_tracker(tracker, state)

        assert updated.portfolio.cash == 75_000
        assert len(updated.portfolio.positions) == 2
        assert updated.portfolio.positions["AAPL"].shares == 50


class TestRunner:
    @patch("src.paper_trading.runner.run_hedge_fund")
    def test_run_cycle_applies_trades(self, mock_run):
        from src.paper_trading.runner import PaperTradingRunner

        mock_run.return_value = {
            "data": {
                "portfolio_output": {
                    "positions": [
                        {"ticker": "AAPL", "action": "buy", "quantity": 10,
                         "confidence": 70, "reasoning": "Strong fundamentals"},
                    ],
                },
                "current_prices": {"AAPL": 200.0},
            },
        }

        state = _make_state()
        runner = PaperTradingRunner(state=state)
        updated = runner.run_cycle()

        assert updated.run_count == 1
        assert updated.last_run is not None
        assert len(updated.trades) == 1
        assert updated.trades[0].ticker == "AAPL"
        assert updated.trades[0].action == "buy"
        assert "AAPL" in updated.portfolio.positions
        assert updated.portfolio.cash < 100_000

    @patch("src.paper_trading.runner.run_hedge_fund")
    def test_run_cycle_no_trades(self, mock_run):
        from src.paper_trading.runner import PaperTradingRunner

        mock_run.return_value = {
            "data": {
                "portfolio_output": {
                    "positions": [
                        {"ticker": "AAPL", "action": "hold", "quantity": 0,
                         "confidence": 50, "reasoning": "Hold position"},
                    ],
                },
                "current_prices": {"AAPL": 200.0},
            },
        }

        state = _make_state()
        runner = PaperTradingRunner(state=state)
        updated = runner.run_cycle()

        assert updated.run_count == 1
        assert len(updated.trades) == 0
        assert updated.portfolio.cash == 100_000

    @patch("src.paper_trading.runner.run_hedge_fund")
    def test_run_cycle_preserves_existing_trades(self, mock_run):
        from src.paper_trading.runner import PaperTradingRunner

        mock_run.return_value = {
            "data": {
                "portfolio_output": {"positions": []},
                "current_prices": {"AAPL": 200.0},
            },
        }

        state = _make_state(run_count=2)
        state.trades = [
            Trade(date=date(2024, 1, 10), ticker="AAPL", action="buy",
                  quantity=5, price=190.0, total_value=950.0),
        ]

        runner = PaperTradingRunner(state=state)
        updated = runner.run_cycle()

        assert updated.run_count == 3
        assert len(updated.trades) == 1  # existing trade preserved

    @patch("src.paper_trading.runner.run_hedge_fund")
    def test_run_cycle_takes_snapshot(self, mock_run):
        from src.paper_trading.runner import PaperTradingRunner

        mock_run.return_value = {
            "data": {
                "portfolio_output": {"positions": []},
                "current_prices": {"AAPL": 200.0},
            },
        }

        state = _make_state()
        runner = PaperTradingRunner(state=state)
        updated = runner.run_cycle()

        assert len(updated.snapshots) == 1
        assert updated.snapshots[0].total_value == 100_000


class TestCLI:
    def test_run_subcommand_parses(self):
        from src.paper_trader import parse_args
        import sys

        with patch.object(sys, "argv", [
            "paper_trader", "run", "--ticker", "AAPL,MSFT",
            "--cash", "50000", "--use-llm",
        ]):
            args = parse_args()
            assert args.command == "run"
            assert args.ticker == "AAPL,MSFT"
            assert args.cash == 50000
            assert args.use_llm is True

    def test_status_subcommand_parses(self):
        from src.paper_trader import parse_args
        import sys

        with patch.object(sys, "argv", [
            "paper_trader", "status", "--state-file", "my_portfolio.json",
        ]):
            args = parse_args()
            assert args.command == "status"
            assert args.state_file == "my_portfolio.json"

    def test_reset_subcommand_parses(self):
        from src.paper_trader import parse_args
        import sys

        with patch.object(sys, "argv", [
            "paper_trader", "reset", "--ticker", "NVDA", "--cash", "200000",
        ]):
            args = parse_args()
            assert args.command == "reset"
            assert args.ticker == "NVDA"
            assert args.cash == 200000
