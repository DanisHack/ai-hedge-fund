"""Tests for backtest result export (JSON & CSV)."""
from __future__ import annotations

import csv
import json
from datetime import date

import pytest

from src.backtest.export import export_csv, export_json, export_results
from src.backtest.models import (
    BacktestResult,
    BenchmarkResult,
    HoldingDetail,
    PerformanceMetrics,
    PortfolioSnapshot,
    Trade,
)


def _make_result() -> BacktestResult:
    """Build a minimal BacktestResult for testing."""
    return BacktestResult(
        tickers=["AAPL", "MSFT"],
        start_date=date(2025, 1, 1),
        end_date=date(2025, 2, 1),
        frequency="weekly",
        initial_cash=100_000.0,
        final_value=105_000.0,
        metrics=PerformanceMetrics(
            total_return_pct=5.0,
            annualized_return_pct=60.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=-3.5,
            max_drawdown_start=date(2025, 1, 10),
            max_drawdown_end=date(2025, 1, 15),
            volatility_annual_pct=12.0,
            calmar_ratio=4.5,
            total_trades=3,
            winning_trades=2,
            losing_trades=1,
            win_rate_pct=66.67,
            avg_win_pct=4.0,
            avg_loss_pct=-2.0,
            profit_factor=2.0,
        ),
        benchmark=BenchmarkResult(
            ticker="SPY",
            start_price=450.0,
            end_price=459.0,
            total_return_pct=2.0,
            annualized_return_pct=24.0,
            sharpe_ratio=1.0,
            max_drawdown_pct=-2.0,
        ),
        trades=[
            Trade(
                date=date(2025, 1, 3),
                ticker="AAPL",
                action="buy",
                quantity=10,
                price=150.0,
                total_value=1500.0,
            ),
            Trade(
                date=date(2025, 1, 17),
                ticker="MSFT",
                action="sell",
                quantity=5,
                price=400.0,
                total_value=2000.0,
            ),
        ],
        snapshots=[
            PortfolioSnapshot(
                date=date(2025, 1, 3),
                cash=98_500.0,
                total_value=100_000.0,
                daily_return=None,
                holdings={
                    "AAPL": HoldingDetail(
                        shares=10, avg_cost=150.0, current_price=150.0,
                        market_value=1500.0, unrealized_pnl=0.0,
                    ),
                },
            ),
            PortfolioSnapshot(
                date=date(2025, 1, 10),
                cash=98_500.0,
                total_value=100_500.0,
                daily_return=0.5,
                holdings={
                    "AAPL": HoldingDetail(
                        shares=10, avg_cost=150.0, current_price=155.0,
                        market_value=1550.0, unrealized_pnl=50.0,
                    ),
                },
            ),
        ],
    )


class TestExportJson:
    def test_export_json(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "results.json")
        export_json(result, path)

        with open(path) as f:
            data = json.load(f)

        assert data["tickers"] == ["AAPL", "MSFT"]
        assert data["start_date"] == "2025-01-01"
        assert data["end_date"] == "2025-02-01"
        assert data["frequency"] == "weekly"
        assert data["initial_cash"] == 100_000.0
        assert data["final_value"] == 105_000.0
        assert data["metrics"]["sharpe_ratio"] == 1.5
        assert data["benchmark"]["ticker"] == "SPY"
        assert len(data["trades"]) == 2
        assert len(data["snapshots"]) == 2

    def test_export_json_roundtrip(self, tmp_path):
        """Verify dates serialize as ISO strings and fields survive roundtrip."""
        result = _make_result()
        path = str(tmp_path / "roundtrip.json")
        export_json(result, path)

        with open(path) as f:
            data = json.load(f)

        # Dates should be ISO strings
        assert data["start_date"] == "2025-01-01"
        assert data["end_date"] == "2025-02-01"
        assert data["trades"][0]["date"] == "2025-01-03"
        assert data["snapshots"][0]["date"] == "2025-01-03"

        # Roundtrip back to model
        restored = BacktestResult(**data)
        assert restored.tickers == result.tickers
        assert restored.final_value == result.final_value
        assert len(restored.trades) == len(result.trades)
        assert len(restored.snapshots) == len(result.snapshots)

    def test_export_json_none_metrics(self, tmp_path):
        """Optional metric fields serialize as null."""
        result = BacktestResult(
            tickers=["AAPL"],
            start_date=date(2025, 1, 1),
            end_date=date(2025, 2, 1),
            frequency="weekly",
            initial_cash=100_000.0,
            final_value=100_000.0,
            metrics=PerformanceMetrics(
                total_return_pct=0.0,
                annualized_return_pct=0.0,
                max_drawdown_pct=0.0,
            ),
        )
        path = str(tmp_path / "none_metrics.json")
        export_json(result, path)

        with open(path) as f:
            data = json.load(f)

        assert data["metrics"]["sharpe_ratio"] is None
        assert data["benchmark"] is None


class TestExportCsv:
    def test_export_csv_sections(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "results.csv")
        export_csv(result, path)

        with open(path) as f:
            content = f.read()

        assert "# Summary" in content
        assert "# Metrics" in content
        assert "# Trades" in content
        assert "# Snapshots" in content

    def test_export_csv_summary_headers(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "results.csv")
        export_csv(result, path)

        with open(path) as f:
            lines = f.readlines()

        # Line 0: "# Summary", Line 1: headers
        header_line = lines[1].strip()
        assert "tickers" in header_line
        assert "frequency" in header_line
        assert "initial_cash" in header_line
        assert "final_value" in header_line
        assert "total_return_pct" in header_line

    def test_export_csv_trades_section(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "results.csv")
        export_csv(result, path)

        with open(path) as f:
            content = f.read()

        assert "AAPL" in content
        assert "buy" in content
        assert "MSFT" in content
        assert "sell" in content

    def test_export_csv_snapshot_ticker_columns(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "results.csv")
        export_csv(result, path)

        with open(path) as f:
            content = f.read()

        assert "holdings_AAPL" in content
        assert "holdings_MSFT" in content

    def test_export_csv_benchmark_columns(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "results.csv")
        export_csv(result, path)

        with open(path) as f:
            content = f.read()

        assert "benchmark_ticker" in content
        assert "benchmark_total_return_pct" in content

    def test_export_csv_parseable(self, tmp_path):
        """CSV can be parsed by csv.reader."""
        result = _make_result()
        path = str(tmp_path / "results.csv")
        export_csv(result, path)

        with open(path) as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Sections + headers + data + blank lines
        assert len(rows) > 10


class TestExportResults:
    def test_dispatch_json(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "out.json")
        export_results(result, path)

        with open(path) as f:
            data = json.load(f)
        assert data["tickers"] == ["AAPL", "MSFT"]

    def test_dispatch_csv(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "out.csv")
        export_results(result, path)

        with open(path) as f:
            content = f.read()
        assert "# Summary" in content

    def test_unknown_extension(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "out.xlsx")
        with pytest.raises(ValueError, match="Unsupported export format"):
            export_results(result, path)

    def test_case_insensitive_extension(self, tmp_path):
        result = _make_result()
        path = str(tmp_path / "out.JSON")
        export_results(result, path)

        with open(path) as f:
            data = json.load(f)
        assert data["tickers"] == ["AAPL", "MSFT"]
