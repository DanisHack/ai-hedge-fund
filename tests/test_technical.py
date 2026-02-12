"""Tests for technical analyst agent."""
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from src.agents.technical import technical_agent, _compute_rsi
from src.data.models import Price


def _make_state(tickers=("AAPL",), start_date="2024-01-01", end_date="2024-06-01"):
    return {
        "messages": [],
        "data": {"tickers": list(tickers), "start_date": start_date, "end_date": end_date},
        "metadata": {"show_reasoning": False},
    }


def _make_prices(closes, volumes=None):
    if volumes is None:
        volumes = [1_000_000] * len(closes)
    return [
        Price(
            open=c, high=c + 1, low=c - 1, close=c, volume=v,
            timestamp=datetime(2024, 1, 1) + timedelta(days=i),
        )
        for i, (c, v) in enumerate(zip(closes, volumes))
    ]


class TestComputeRsi:
    def test_all_gains_rsi_100(self):
        closes = np.array([float(i) for i in range(100, 116)])  # 16 values, monotonically rising
        assert _compute_rsi(closes, period=14) == 100.0

    def test_all_losses_rsi_0(self):
        closes = np.array([float(i) for i in range(116, 100, -1)])  # 16 values, monotonically falling
        assert _compute_rsi(closes, period=14) == pytest.approx(0.0)

    def test_mixed_returns_value(self):
        closes = np.array([100.0, 102.0, 101.0, 103.0, 102.0, 104.0, 103.0, 105.0,
                           104.0, 106.0, 105.0, 107.0, 106.0, 108.0, 107.0])
        rsi = _compute_rsi(closes, period=14)
        assert rsi is not None
        assert 0 < rsi < 100

    def test_insufficient_data_returns_none(self):
        closes = np.array([100.0] * 14)  # Needs period+1=15, only 14
        assert _compute_rsi(closes, period=14) is None


class TestTechnicalAgent:
    @patch("src.agents.technical.get_prices")
    def test_insufficient_bars_returns_neutral(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 30)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10

    @patch("src.agents.technical.get_prices")
    def test_insufficient_bars_still_returns_latest_price(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0, 105.0, 110.0])
        result = technical_agent(_make_state())
        assert result["data"]["current_prices"]["AAPL"] == 110.0

    @patch("src.agents.technical.get_prices")
    def test_empty_prices_returns_neutral(self, mock_prices):
        mock_prices.return_value = []
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10
        assert "AAPL" not in result["data"]["current_prices"]

    @patch("src.agents.technical.get_prices")
    def test_strong_bullish_signal(self, mock_prices):
        # Trending upward: SMA20 > SMA50, RSI in neutral zone, rising volume, price > SMA50
        closes = [100.0 + i * 0.5 for i in range(60)]  # 100 to 129.5
        volumes = [500_000] * 50 + [1_000_000] * 10  # Volume spike in last 10 days
        mock_prices.return_value = _make_prices(closes, volumes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "bullish"

    @patch("src.agents.technical.get_prices")
    def test_strong_bearish_signal(self, mock_prices):
        # Trending downward: SMA20 < SMA50, RSI high (overbought), flat volume, price < SMA50
        closes = [200.0 - i * 1.0 for i in range(60)]  # 200 down to 141
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "bearish"

    @patch("src.agents.technical.get_prices")
    def test_neutral_signal_mid_range(self, mock_prices):
        # Slowly rising: SMA20>SMA50 (+2), RSI=100 overbought (+0), flat vol (+0),
        # price>SMA50 (+1) → 3/6=0.5 → neutral
        closes = [99.0 + i * 0.03 for i in range(60)]
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "neutral"

    @patch("src.agents.technical.get_prices")
    def test_current_prices_populated(self, mock_prices):
        closes = [100.0 + i * 0.5 for i in range(60)]
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state(tickers=("AAPL", "MSFT")))
        assert "AAPL" in result["data"]["current_prices"]
        assert "MSFT" in result["data"]["current_prices"]

    @patch("src.agents.technical.get_prices")
    def test_api_exception_returns_neutral_confidence_0(self, mock_prices):
        mock_prices.side_effect = Exception("API error")
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 0

    @patch("src.agents.technical.get_prices")
    def test_output_structure(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 60)
        result = technical_agent(_make_state())
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "analyst_signals" in result["data"]
        assert "technical_analyst" in result["data"]["analyst_signals"]
        assert "current_prices" in result["data"]

    @patch("src.agents.technical.get_prices")
    def test_exactly_50_bars_is_sufficient(self, mock_prices):
        closes = [100.0 + i * 0.3 for i in range(50)]
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        # With 50 bars, the agent should run full analysis (not the early return)
        assert signals[0]["confidence"] != 10 or "Insufficient" not in signals[0]["reasoning"]
