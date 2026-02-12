"""Tests for risk manager agent."""
from datetime import datetime, timedelta
from unittest.mock import patch

from src.agents.risk_manager import risk_manager_agent, _collect_signals_for_ticker
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
