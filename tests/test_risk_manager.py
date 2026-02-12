"""Tests for risk manager agent."""
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.agents.risk_manager import (
    _build_correlation_groups,
    _collect_signals_for_ticker,
    _compute_correlation_matrix,
    _correlation_adjusted_position_size,
    risk_manager_agent,
)
from src.data.models import Price


def _make_state(tickers=("AAPL",), analyst_signals=None, portfolio=None,
                start_date="2024-01-01", end_date="2024-06-01"):
    if portfolio is None:
        portfolio = {"cash": 100_000, "positions": {}, "total_value": 100_000}
    return {
        "messages": [],
        "data": {
            "tickers": list(tickers),
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": analyst_signals or {},
            "portfolio": portfolio,
        },
        "metadata": {"show_reasoning": False},
    }


def _make_signal(ticker, signal, confidence, agent_id="agent_1"):
    return {"agent_id": agent_id, "ticker": ticker, "signal": signal, "confidence": confidence, "reasoning": "test"}


def _make_prices(closes):
    return [
        Price(open=c, high=c + 1, low=c - 1, close=c, volume=1_000_000,
              timestamp=datetime(2024, 1, 1) + timedelta(days=i))
        for i, c in enumerate(closes)
    ]


class TestCollectSignals:
    def test_collects_from_multiple_agents(self):
        signals = {
            "agent_a": [_make_signal("AAPL", "bullish", 80)],
            "agent_b": [_make_signal("AAPL", "neutral", 50)],
            "agent_c": [_make_signal("AAPL", "bearish", 70)],
        }
        result = _collect_signals_for_ticker("AAPL", signals)
        assert len(result) == 3

    def test_filters_by_ticker(self):
        signals = {
            "agent_a": [_make_signal("AAPL", "bullish", 80), _make_signal("MSFT", "bearish", 60)],
        }
        result = _collect_signals_for_ticker("AAPL", signals)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_empty_signals(self):
        assert _collect_signals_for_ticker("AAPL", {}) == []


class TestRiskManagerAgent:
    @patch("src.agents.risk_manager.get_prices")
    def test_bullish_consensus(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 25)
        signals = {
            "agent_a": [_make_signal("AAPL", "bullish", 80)],
            "agent_b": [_make_signal("AAPL", "bullish", 70)],
            "agent_c": [_make_signal("AAPL", "bearish", 60)],
        }
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert adjusted[0]["signal"] == "bullish"

    @patch("src.agents.risk_manager.get_prices")
    def test_bearish_consensus(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 25)
        signals = {
            "agent_a": [_make_signal("AAPL", "bearish", 80)],
            "agent_b": [_make_signal("AAPL", "bearish", 70)],
            "agent_c": [_make_signal("AAPL", "bullish", 60)],
        }
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert adjusted[0]["signal"] == "bearish"

    @patch("src.agents.risk_manager.get_prices")
    def test_tie_is_neutral(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 25)
        signals = {
            "agent_a": [_make_signal("AAPL", "bullish", 80)],
            "agent_b": [_make_signal("AAPL", "bearish", 80)],
        }
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert adjusted[0]["signal"] == "neutral"

    @patch("src.agents.risk_manager.get_prices")
    def test_no_signals_returns_neutral_confidence_0(self, mock_prices):
        mock_prices.return_value = []
        result = risk_manager_agent(_make_state(analyst_signals={}))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert adjusted[0]["signal"] == "neutral"
        assert adjusted[0]["confidence"] == 0
        assert adjusted[0]["max_position_size"] == 0

    @patch("src.agents.risk_manager.get_prices")
    def test_volatility_penalty_applied(self, mock_prices):
        # Create prices with ~5% daily volatility (alternating up/down by 5%)
        closes = []
        price = 100.0
        for i in range(25):
            closes.append(price)
            price = price * (1.05 if i % 2 == 0 else 0.95)
        mock_prices.return_value = _make_prices(closes)
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        # Confidence should be reduced from 80
        assert adjusted[0]["confidence"] < 80
        assert "volatility" in adjusted[0]["reasoning"].lower()

    @patch("src.agents.risk_manager.get_prices")
    def test_volatility_penalty_capped_at_30(self, mock_prices):
        # Very high volatility
        closes = []
        price = 100.0
        for i in range(25):
            closes.append(price)
            price = price * (1.20 if i % 2 == 0 else 0.80)
        mock_prices.return_value = _make_prices(closes)
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        # Min confidence = 80 - 30 = 50
        assert adjusted[0]["confidence"] >= 50

    @patch("src.agents.risk_manager.get_prices")
    def test_no_volatility_penalty_when_low_vol(self, mock_prices):
        # Flat prices → 0 volatility
        mock_prices.return_value = _make_prices([100.0] * 25)
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert adjusted[0]["confidence"] == 80
        assert "volatility" not in adjusted[0]["reasoning"].lower()

    @patch("src.agents.risk_manager.get_prices")
    def test_exposure_penalty_when_fully_invested_bullish(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 25)
        portfolio = {
            "cash": 5_000,
            "positions": {"MSFT": {"shares": 100, "value": 95_000}},
            "total_value": 100_000,
        }
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals, portfolio=portfolio))
        adjusted = result["data"]["risk_adjusted_signals"]
        # 95% exposure > 90% max, bullish → -50 penalty
        assert adjusted[0]["confidence"] <= 30
        assert "exposure" in adjusted[0]["reasoning"].lower()

    @patch("src.agents.risk_manager.get_prices")
    def test_no_exposure_penalty_for_bearish(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 25)
        portfolio = {
            "cash": 5_000,
            "positions": {"MSFT": {"shares": 100, "value": 95_000}},
            "total_value": 100_000,
        }
        signals = {"agent_a": [_make_signal("AAPL", "bearish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals, portfolio=portfolio))
        adjusted = result["data"]["risk_adjusted_signals"]
        # Bearish → no exposure penalty
        assert "exposure" not in adjusted[0]["reasoning"].lower()

    @patch("src.agents.risk_manager.get_prices")
    def test_max_position_size_calculated(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 25)
        portfolio = {"cash": 200_000, "positions": {}, "total_value": 200_000}
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals, portfolio=portfolio))
        adjusted = result["data"]["risk_adjusted_signals"]
        # 25% of 200k = 50k
        assert adjusted[0]["max_position_size"] == 50_000

    @patch("src.agents.risk_manager.get_prices")
    def test_confidence_floored_at_0(self, mock_prices):
        # Very high volatility + exposure penalty → should not go negative
        closes = []
        price = 100.0
        for i in range(25):
            closes.append(price)
            price = price * (1.20 if i % 2 == 0 else 0.80)
        mock_prices.return_value = _make_prices(closes)
        portfolio = {
            "cash": 5_000,
            "positions": {"MSFT": {"shares": 100, "value": 95_000}},
            "total_value": 100_000,
        }
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 20)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals, portfolio=portfolio))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert adjusted[0]["confidence"] >= 0

    @patch("src.agents.risk_manager.get_prices")
    def test_prices_exception_handled(self, mock_prices):
        mock_prices.side_effect = Exception("API error")
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert "Could not compute volatility" in adjusted[0]["reasoning"]
        # Should still produce a result
        assert adjusted[0]["signal"] == "bullish"

    @patch("src.agents.risk_manager.get_prices")
    def test_insufficient_price_bars_for_volatility(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 5)  # < 20 lookback
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}
        result = risk_manager_agent(_make_state(analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        # No volatility penalty since < 20 bars
        assert adjusted[0]["confidence"] == 80

    @patch("src.agents.risk_manager.get_prices")
    def test_output_structure(self, mock_prices):
        mock_prices.return_value = []
        result = risk_manager_agent(_make_state(analyst_signals={}))
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "risk_adjusted_signals" in result["data"]
        adjusted = result["data"]["risk_adjusted_signals"]
        assert isinstance(adjusted, list)
        entry = adjusted[0]
        assert all(k in entry for k in ("ticker", "signal", "confidence", "reasoning", "max_position_size"))

    @patch("src.agents.risk_manager.get_prices")
    def test_multiple_tickers(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 25)
        signals = {
            "agent_a": [_make_signal("AAPL", "bullish", 80), _make_signal("MSFT", "bearish", 60)],
        }
        result = risk_manager_agent(_make_state(tickers=("AAPL", "MSFT"), analyst_signals=signals))
        adjusted = result["data"]["risk_adjusted_signals"]
        assert len(adjusted) == 2
        assert {a["ticker"] for a in adjusted} == {"AAPL", "MSFT"}


class TestCorrelationMatrix:
    @patch("src.agents.risk_manager.get_prices")
    def test_perfectly_correlated_stocks(self, mock_prices):
        closes = [100 + i for i in range(65)]
        mock_prices.return_value = _make_prices(closes)

        result = _compute_correlation_matrix(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")

        assert ("AAPL", "MSFT") in result
        assert result[("AAPL", "MSFT")] == pytest.approx(1.0, abs=0.01)

    @patch("src.agents.risk_manager.get_prices")
    def test_uncorrelated_stocks(self, mock_prices):
        def side_effect(ticker, *args, **kwargs):
            if ticker == "AAPL":
                return _make_prices([100 + i for i in range(65)])
            else:
                return _make_prices([100 + (-1) ** i * i * 0.5 for i in range(65)])

        mock_prices.side_effect = side_effect

        result = _compute_correlation_matrix(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")

        assert ("AAPL", "MSFT") in result
        assert result[("AAPL", "MSFT")] < 0.7

    @patch("src.agents.risk_manager.get_prices")
    def test_insufficient_data_excluded(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 10)

        result = _compute_correlation_matrix(["AAPL", "MSFT"], "2024-01-01", "2024-06-01")

        assert result == {}

    @patch("src.agents.risk_manager.get_prices")
    def test_api_error_handled_gracefully(self, mock_prices):
        def side_effect(ticker, *args, **kwargs):
            if ticker == "AAPL":
                raise Exception("API error")
            return _make_prices([100 + i for i in range(65)])

        mock_prices.side_effect = side_effect

        result = _compute_correlation_matrix(["AAPL", "MSFT", "GOOG"], "2024-01-01", "2024-06-01")

        # AAPL excluded; MSFT-GOOG still computed
        assert ("AAPL", "MSFT") not in result
        assert ("MSFT", "GOOG") in result


class TestCorrelationGroups:
    def test_builds_group_from_high_correlation(self):
        correlations = {
            ("AAPL", "MSFT"): 0.85,
            ("MSFT", "AAPL"): 0.85,
        }
        groups = _build_correlation_groups(["AAPL", "MSFT", "GOOG"], correlations)
        assert len(groups) == 1
        assert groups[0] == {"AAPL", "MSFT"}

    def test_transitive_grouping(self):
        correlations = {
            ("A", "B"): 0.8, ("B", "A"): 0.8,
            ("B", "C"): 0.8, ("C", "B"): 0.8,
            ("A", "C"): 0.3, ("C", "A"): 0.3,
        }
        groups = _build_correlation_groups(["A", "B", "C"], correlations)
        assert len(groups) == 1
        assert groups[0] == {"A", "B", "C"}

    def test_no_groups_when_all_below_threshold(self):
        correlations = {
            ("AAPL", "GOOG"): 0.3,
            ("GOOG", "AAPL"): 0.3,
        }
        groups = _build_correlation_groups(["AAPL", "GOOG"], correlations)
        assert groups == []


class TestCorrelationAdjustedPositionSize:
    def test_reduces_size_when_group_near_cap(self):
        portfolio = {
            "cash": 60_000,
            "positions": {"MSFT": {"shares": 100, "avg_cost": 350.0}},
            "total_value": 100_000,
        }
        correlation_groups = [{"AAPL", "MSFT"}]

        adjusted, reason = _correlation_adjusted_position_size(
            "AAPL", 25_000, portfolio, correlation_groups,
        )

        # Group cap = 100k * 0.40 = 40k. MSFT exposure = 100*350 = 35k. Remaining = 5k.
        assert adjusted == 5_000
        assert "MSFT" in reason

    def test_no_reduction_when_no_correlated_positions(self):
        portfolio = {
            "cash": 100_000,
            "positions": {},
            "total_value": 100_000,
        }
        adjusted, reason = _correlation_adjusted_position_size(
            "AAPL", 25_000, portfolio, [],
        )
        assert adjusted == 25_000
        assert reason is None

    def test_zero_remaining_capacity(self):
        portfolio = {
            "cash": 55_000,
            "positions": {"MSFT": {"shares": 100, "avg_cost": 450.0}},
            "total_value": 100_000,
        }
        correlation_groups = [{"AAPL", "MSFT"}]

        adjusted, reason = _correlation_adjusted_position_size(
            "AAPL", 25_000, portfolio, correlation_groups,
        )

        # Group cap = 40k, MSFT exposure = 45k > 40k → remaining = 0
        assert adjusted == 0

    @patch("src.agents.risk_manager.get_prices")
    def test_end_to_end_reduces_for_correlated_buy(self, mock_prices):
        # Both tickers have identical returns → correlation ~1.0
        closes = [100 + i for i in range(65)]
        mock_prices.return_value = _make_prices(closes)

        portfolio = {
            "cash": 60_000,
            "positions": {"MSFT": {"shares": 100, "avg_cost": 350.0}},
            "total_value": 100_000,
        }
        signals = {"agent_a": [_make_signal("AAPL", "bullish", 80)]}

        result = risk_manager_agent(_make_state(
            tickers=("AAPL",),
            analyst_signals=signals,
            portfolio=portfolio,
        ))
        adjusted = result["data"]["risk_adjusted_signals"]

        # max_position_size should be reduced from 25k due to correlation with MSFT
        assert adjusted[0]["max_position_size"] < 25_000
        assert "correlation" in adjusted[0]["reasoning"].lower()
