"""Tests for technical analyst agent."""
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pytest

from src.agents.technical import (
    _compute_adx,
    _compute_bollinger,
    _compute_macd,
    _compute_rsi,
    technical_agent,
)
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


class TestComputeMacd:
    def test_insufficient_data_returns_none(self):
        closes = np.array([100.0] * 30)  # needs 26+9=35
        assert _compute_macd(closes) is None

    def test_sufficient_data_returns_tuple(self):
        closes = np.array([100.0 + i * 0.5 for i in range(60)])
        result = _compute_macd(closes)
        assert result is not None
        macd_line, signal_line, histogram = result
        assert isinstance(macd_line, float)
        assert isinstance(signal_line, float)
        assert histogram == pytest.approx(macd_line - signal_line)

    def test_uptrend_macd_positive(self):
        closes = np.array([100.0 + i * 1.0 for i in range(60)])
        macd_line, _, _ = _compute_macd(closes)
        assert macd_line > 0

    def test_downtrend_macd_negative(self):
        closes = np.array([200.0 - i * 1.0 for i in range(60)])
        macd_line, _, _ = _compute_macd(closes)
        assert macd_line < 0

    def test_flat_prices_macd_near_zero(self):
        closes = np.array([100.0] * 60)
        macd_line, signal_line, histogram = _compute_macd(closes)
        assert abs(macd_line) < 0.01
        assert abs(signal_line) < 0.01


class TestComputeBollinger:
    def test_insufficient_data_returns_none(self):
        closes = np.array([100.0] * 15)  # needs 20
        assert _compute_bollinger(closes) is None

    def test_sufficient_data_returns_tuple(self):
        closes = np.array([100.0 + i * 0.1 for i in range(30)])
        result = _compute_bollinger(closes)
        assert result is not None
        upper, middle, lower, pct_b = result
        assert upper > middle > lower

    def test_constant_prices_pct_b_half(self):
        closes = np.array([100.0] * 20)
        result = _compute_bollinger(closes)
        assert result is not None
        _, _, _, pct_b = result
        assert pct_b == pytest.approx(0.5)

    def test_rising_prices_pct_b_above_half(self):
        closes = np.array([100.0 + i * 2.0 for i in range(30)])
        result = _compute_bollinger(closes)
        _, _, _, pct_b = result
        assert pct_b > 0.5

    def test_falling_prices_pct_b_below_half(self):
        closes = np.array([200.0 - i * 2.0 for i in range(30)])
        result = _compute_bollinger(closes)
        _, _, _, pct_b = result
        assert pct_b < 0.5


class TestComputeAdx:
    def test_insufficient_data_returns_none(self):
        n = 20  # needs 28 (2*14)
        assert _compute_adx(
            np.array([101.0] * n), np.array([99.0] * n), np.array([100.0] * n),
        ) is None

    def test_sufficient_data_returns_float(self):
        n = 60
        closes = np.array([100.0 + i * 0.5 for i in range(n)])
        highs = closes + 1.0
        lows = closes - 1.0
        adx = _compute_adx(highs, lows, closes)
        assert adx is not None
        assert 0 <= adx <= 100

    def test_strong_trend_high_adx(self):
        n = 60
        closes = np.array([100.0 + i * 2.0 for i in range(n)])
        highs = closes + 0.5
        lows = closes - 0.5
        adx = _compute_adx(highs, lows, closes)
        assert adx is not None
        assert adx > 25

    def test_ranging_market_lower_adx(self):
        n = 60
        closes = np.array([100.0 + (1 if i % 2 == 0 else -1) for i in range(n)])
        highs = closes + 1.0
        lows = closes - 1.0
        adx = _compute_adx(highs, lows, closes)
        assert adx is not None
        # Ranging market should have lower ADX than a strong trend
        assert adx < 50


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
        # Rising trend with pullbacks: keeps RSI in neutral zone, BB within bands
        # SMA20>SMA50(+2), RSI neutral(+1), vol spike(+1), price>SMA50(+1),
        # MACD bullish(+2), BB mid(+1), ADX strong(+1) → 9/11=0.82 → bullish
        closes = []
        price = 100.0
        for i in range(60):
            price += 1.3 if i % 3 != 2 else -1.2
            closes.append(price)
        volumes = [500_000] * 50 + [1_000_000] * 10
        mock_prices.return_value = _make_prices(closes, volumes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "bullish"

    @patch("src.agents.technical.get_prices")
    def test_strong_bearish_signal(self, mock_prices):
        # 45 flat bars then mild decline: SMA20<SMA50, MACD bearish,
        # RSI neutral (31), BB contrarian (+2), ADX weak → 3/11=0.27 → bearish
        closes = [150.0 + (0.3 if i % 2 == 0 else -0.3) for i in range(45)]
        price = closes[-1]
        for i in range(15):
            price += 0.25 if i % 2 == 0 else -0.55
            closes.append(price)
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert signals[0]["signal"] == "bearish"

    @patch("src.agents.technical.get_prices")
    def test_neutral_signal_mid_range(self, mock_prices):
        # Very slowly rising: MACD slightly bullish (+2), BB upper (+0), ADX weak (+0)
        # SMA20>SMA50 (+2), RSI=100 overbought (+0), flat vol (+0), price>SMA50 (+1)
        # total ~5/11=0.45 → neutral
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

    @patch("src.agents.technical.get_prices")
    def test_macd_in_reasoning(self, mock_prices):
        closes = [100.0 + i * 0.5 for i in range(60)]
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert "MACD" in signals[0]["reasoning"]

    @patch("src.agents.technical.get_prices")
    def test_bollinger_in_reasoning(self, mock_prices):
        closes = [100.0 + i * 0.5 for i in range(60)]
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert "Bollinger" in signals[0]["reasoning"]

    @patch("src.agents.technical.get_prices")
    def test_adx_in_reasoning(self, mock_prices):
        closes = [100.0 + i * 0.5 for i in range(60)]
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert "ADX" in signals[0]["reasoning"]

    @patch("src.agents.technical.get_prices")
    def test_macd_bearish_in_downtrend(self, mock_prices):
        closes = [200.0 - i * 1.0 for i in range(60)]
        mock_prices.return_value = _make_prices(closes)
        result = technical_agent(_make_state())
        signals = result["data"]["analyst_signals"]["technical_analyst"]
        assert "MACD bearish" in signals[0]["reasoning"]
