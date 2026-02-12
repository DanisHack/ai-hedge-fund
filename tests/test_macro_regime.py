"""Tests for macro regime / market environment analyst agent."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from src.agents.macro_regime import (
    AGENT_ID,
    MarketRegime,
    _analyze_ticker,
    _compute_regime,
    _get_ticker_sector,
    macro_regime_agent,
)
from src.data.models import Price, SignalType


def _make_state(tickers=("AAPL",), start_date="2024-01-01", end_date="2024-06-01"):
    return {
        "messages": [],
        "data": {
            "tickers": list(tickers),
            "start_date": start_date,
            "end_date": end_date,
        },
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


# ── Ticker sector lookup ──────────────────────────────────────────────


class TestGetTickerSector:
    def test_known_override_tech(self):
        assert _get_ticker_sector("AAPL") == "technology"

    def test_known_override_financials(self):
        assert _get_ticker_sector("JPM") == "financials"

    def test_known_override_energy(self):
        assert _get_ticker_sector("XOM") == "energy"

    @patch("src.agents.macro_regime.get_company_details")
    def test_sic_code_fallback(self, mock_details):
        details = MagicMock()
        details.sic_code = "7372"  # prefix "73" -> technology
        mock_details.return_value = details
        assert _get_ticker_sector("UNKNOWN_TICKER") == "technology"

    @patch("src.agents.macro_regime.get_company_details")
    def test_no_details_returns_none(self, mock_details):
        mock_details.return_value = None
        assert _get_ticker_sector("UNKNOWN_TICKER") is None

    @patch("src.agents.macro_regime.get_company_details")
    def test_api_error_returns_none(self, mock_details):
        mock_details.side_effect = Exception("API error")
        assert _get_ticker_sector("UNKNOWN_TICKER") is None


# ── Regime computation ────────────────────────────────────────────────


class TestComputeRegime:
    @patch("src.agents.macro_regime.get_prices")
    def test_insufficient_spy_data(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0] * 30)  # <50 bars
        regime = _compute_regime("2024-01-01", "2024-06-01", ["AAPL"])
        assert regime.spy_volatility is None
        assert regime.spy_above_sma50 is None

    @patch("src.agents.macro_regime.get_prices")
    def test_bullish_spy_trend(self, mock_prices):
        # SPY trending up with enough data for SMA200
        spy_closes = [400.0 + i * 0.5 for i in range(220)]

        def price_for_ticker(ticker, *args, **kwargs):
            if ticker == "SPY":
                return _make_prices(spy_closes)
            # Sector ETFs: cyclicals up more than defensives
            if ticker in ("XLK", "XLF", "XLE", "XLI", "XLY", "XLB"):
                return _make_prices([100.0 + i * 0.8 for i in range(60)])
            return _make_prices([100.0 + i * 0.2 for i in range(60)])

        mock_prices.side_effect = price_for_ticker
        regime = _compute_regime("2024-01-01", "2024-06-01", ["AAPL"])
        assert regime.spy_above_sma50 is True
        assert regime.spy_above_sma200 is True
        assert regime.cyclical_vs_defensive > 0
        assert regime.score > 0

    @patch("src.agents.macro_regime.get_prices")
    def test_bearish_spy_trend(self, mock_prices):
        spy_closes = [500.0 - i * 2.0 for i in range(220)]

        def price_for_ticker(ticker, *args, **kwargs):
            if ticker == "SPY":
                return _make_prices(spy_closes)
            # Defensive ETFs outperforming
            if ticker in ("XLU", "XLP", "XLV", "XLRE", "XLC"):
                return _make_prices([100.0 + i * 0.3 for i in range(60)])
            return _make_prices([100.0 - i * 0.5 for i in range(60)])

        mock_prices.side_effect = price_for_ticker
        regime = _compute_regime("2024-01-01", "2024-06-01", ["AAPL"])
        assert regime.spy_above_sma50 is False

    @patch("src.agents.macro_regime.get_prices")
    def test_breadth_computation(self, mock_prices):
        # All tickers above SMA50 = 100% breadth
        mock_prices.return_value = _make_prices([100.0 + i * 0.5 for i in range(60)])
        regime = _compute_regime("2024-01-01", "2024-06-01", ["AAPL", "MSFT"])
        assert regime.breadth_pct is not None
        assert regime.breadth_pct == 1.0

    @patch("src.agents.macro_regime.get_prices")
    def test_sector_returns_populated(self, mock_prices):
        mock_prices.return_value = _make_prices([100.0 + i * 0.3 for i in range(60)])
        regime = _compute_regime("2024-01-01", "2024-06-01", ["AAPL"])
        # Should have fetched sector ETFs
        assert len(regime.sector_returns) > 0


# ── Per-ticker analysis ───────────────────────────────────────────────


class TestAnalyzeTicker:
    def test_no_regime_returns_neutral(self):
        signal = _analyze_ticker("AAPL", None, "2024-01-01", "2024-06-01")
        assert signal.signal == SignalType.NEUTRAL
        assert signal.confidence == 10

    def test_empty_regime_returns_neutral(self):
        regime = MarketRegime()  # max_score=0
        signal = _analyze_ticker("AAPL", regime, "2024-01-01", "2024-06-01")
        assert signal.signal == SignalType.NEUTRAL
        assert signal.confidence == 10

    def test_bullish_regime_high_confidence(self):
        regime = MarketRegime()
        regime.score = 8
        regime.max_score = 8
        regime.reasons = ["SPY bullish", "Cyclicals leading"]
        regime.sector_returns = {"technology": 0.05, "financials": 0.03}
        regime.leading_sectors = ["technology"]
        regime.lagging_sectors = ["financials"]
        signal = _analyze_ticker("AAPL", regime, "2024-01-01", "2024-06-01")
        assert signal.signal == SignalType.BULLISH
        assert signal.confidence >= 65

    def test_bearish_regime_low_confidence(self):
        regime = MarketRegime()
        regime.score = 1
        regime.max_score = 8
        regime.reasons = ["SPY bearish"]
        regime.sector_returns = {"technology": -0.05}
        regime.leading_sectors = ["utilities"]
        regime.lagging_sectors = ["technology"]
        signal = _analyze_ticker("AAPL", regime, "2024-01-01", "2024-06-01")
        assert signal.signal == SignalType.BEARISH


# ── Full agent integration ────────────────────────────────────────────


class TestMacroRegimeAgent:
    @patch("src.agents.macro_regime.get_company_details")
    @patch("src.agents.macro_regime.get_prices")
    def test_output_structure(self, mock_prices, mock_details):
        mock_prices.return_value = _make_prices([100.0 + i * 0.3 for i in range(60)])
        mock_details.return_value = None
        result = macro_regime_agent(_make_state())
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "analyst_signals" in result["data"]
        assert AGENT_ID in result["data"]["analyst_signals"]

    @patch("src.agents.macro_regime.get_company_details")
    @patch("src.agents.macro_regime.get_prices")
    def test_multiple_tickers(self, mock_prices, mock_details):
        mock_prices.return_value = _make_prices([100.0 + i * 0.5 for i in range(220)])
        mock_details.return_value = None
        result = macro_regime_agent(_make_state(tickers=("AAPL", "MSFT")))
        signals = result["data"]["analyst_signals"][AGENT_ID]
        assert len(signals) == 2
        tickers = {s["ticker"] for s in signals}
        assert tickers == {"AAPL", "MSFT"}

    @patch("src.agents.macro_regime.get_prices")
    def test_api_exception_returns_neutral(self, mock_prices):
        mock_prices.side_effect = Exception("API error")
        result = macro_regime_agent(_make_state())
        signals = result["data"]["analyst_signals"][AGENT_ID]
        assert signals[0]["signal"] == "neutral"
        # Regime computation fails → _analyze_ticker gets regime=None → confidence=10
        assert signals[0]["confidence"] == 10

    @patch("src.agents.macro_regime.get_company_details")
    @patch("src.agents.macro_regime.get_prices")
    def test_signal_fields_present(self, mock_prices, mock_details):
        mock_prices.return_value = _make_prices([100.0 + i * 0.5 for i in range(220)])
        mock_details.return_value = None
        result = macro_regime_agent(_make_state())
        signal = result["data"]["analyst_signals"][AGENT_ID][0]
        assert all(k in signal for k in ("agent_id", "ticker", "signal", "confidence", "reasoning"))
        assert signal["agent_id"] == AGENT_ID

    @patch("src.agents.macro_regime.get_company_details")
    @patch("src.agents.macro_regime.get_prices")
    def test_sector_mentioned_for_known_ticker(self, mock_prices, mock_details):
        mock_prices.return_value = _make_prices([100.0 + i * 0.5 for i in range(220)])
        mock_details.return_value = None
        result = macro_regime_agent(_make_state(tickers=("AAPL",)))
        signal = result["data"]["analyst_signals"][AGENT_ID][0]
        assert "technology" in signal["reasoning"].lower()
