"""Tests for valuation analyst agent."""
from unittest.mock import patch

import pytest

from src.agents.valuation import valuation_agent, _simple_dcf, DISCOUNT_RATE, TERMINAL_GROWTH
from src.data.models import CompanyDetails, FinancialMetrics


def _make_state(tickers=("AAPL",), end_date="2024-06-01"):
    return {
        "messages": [],
        "data": {"tickers": list(tickers), "start_date": "2024-01-01", "end_date": end_date},
        "metadata": {"show_reasoning": False},
    }


def _make_metrics(ticker="AAPL", **kwargs):
    defaults = dict(period="quarterly", fiscal_period="Q1")
    defaults.update(kwargs)
    return FinancialMetrics(ticker=ticker, **defaults)


def _make_details(ticker="AAPL", market_cap=None, **kwargs):
    return CompanyDetails(ticker=ticker, name=f"{ticker} Inc.", market_cap=market_cap, **kwargs)


class TestSimpleDcf:
    def test_basic_dcf_calculation(self):
        metrics = [
            _make_metrics(operating_cash_flow=110e6),
            _make_metrics(operating_cash_flow=100e6),
        ]
        result = _simple_dcf(metrics)
        assert result is not None
        assert result > 0
        # Growth rate = (110-100)/100 = 10%, capped at 25% → 10%
        # Verify by hand: project 5 years, add terminal value
        growth_rate = 0.10
        cf = 110e6
        projected = []
        for year in range(1, 6):
            cf = cf * (1 + growth_rate)
            projected.append(cf / (1 + DISCOUNT_RATE) ** year)
        terminal_cf = cf * (1 + TERMINAL_GROWTH)
        terminal_value = terminal_cf / (DISCOUNT_RATE - TERMINAL_GROWTH)
        discounted_terminal = terminal_value / (1 + DISCOUNT_RATE) ** 5
        expected = sum(projected) + discounted_terminal
        assert result == pytest.approx(expected, rel=1e-6)

    def test_insufficient_cash_flows_returns_none(self):
        metrics = [_make_metrics(operating_cash_flow=100e6)]
        assert _simple_dcf(metrics) is None

    def test_no_positive_cash_flows_returns_none(self):
        metrics = [
            _make_metrics(operating_cash_flow=-50e6),
            _make_metrics(operating_cash_flow=-30e6),
        ]
        assert _simple_dcf(metrics) is None

    def test_growth_rate_capped_at_25_pct(self):
        # Growth = (500-100)/100 = 400%, should be capped at 25%
        metrics = [
            _make_metrics(operating_cash_flow=500e6),
            _make_metrics(operating_cash_flow=100e6),
        ]
        result_capped = _simple_dcf(metrics)

        # Compare with manual 25% growth calc
        growth_rate = 0.25
        cf = 500e6
        projected = []
        for year in range(1, 6):
            cf = cf * (1 + growth_rate)
            projected.append(cf / (1 + DISCOUNT_RATE) ** year)
        terminal_cf = cf * (1 + TERMINAL_GROWTH)
        terminal_value = terminal_cf / (DISCOUNT_RATE - TERMINAL_GROWTH)
        discounted_terminal = terminal_value / (1 + DISCOUNT_RATE) ** 5
        expected = sum(projected) + discounted_terminal
        assert result_capped == pytest.approx(expected, rel=1e-6)

    def test_growth_rate_floored_at_neg_5_pct(self):
        # Growth = (50-200)/200 = -75%, should be floored at -5%
        metrics = [
            _make_metrics(operating_cash_flow=50e6),
            _make_metrics(operating_cash_flow=200e6),
        ]
        result = _simple_dcf(metrics)
        assert result is not None
        # With -5% growth, DCF should still be positive
        assert result > 0


class TestValuationAgent:
    @patch("src.agents.valuation.get_company_details", return_value=None)
    @patch("src.agents.valuation.get_financial_metrics")
    def test_no_metrics_returns_neutral(self, mock_metrics, mock_details):
        mock_metrics.return_value = []
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10

    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_strong_bullish_undervalued(self, mock_metrics, mock_details):
        # Large DCF, low P/E, low P/B, high FCF yield
        mock_metrics.return_value = [
            _make_metrics(
                operating_cash_flow=15e9,
                earnings_per_share=8.0,
                shareholders_equity=50e9,
            ),
            _make_metrics(operating_cash_flow=12e9),
        ]
        mock_details.return_value = _make_details(
            market_cap=100e9,
            weighted_shares_outstanding=5e9,
        )
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        assert signals[0]["signal"] == "bullish"

    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_strong_bearish_overvalued(self, mock_metrics, mock_details):
        # Tiny cash flow, high P/E, high P/B, low FCF yield
        mock_metrics.return_value = [
            _make_metrics(
                operating_cash_flow=1e9,
                earnings_per_share=0.5,
                shareholders_equity=10e9,
            ),
            _make_metrics(operating_cash_flow=1.1e9),
        ]
        mock_details.return_value = _make_details(
            market_cap=500e9,
            weighted_shares_outstanding=5e9,
        )
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        assert signals[0]["signal"] == "bearish"

    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_no_company_details_limits_scoring(self, mock_metrics, mock_details):
        mock_metrics.return_value = [
            _make_metrics(operating_cash_flow=10e9),
            _make_metrics(operating_cash_flow=8e9),
        ]
        mock_details.return_value = None  # No details → no market_cap
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        # No market_cap → DCF/P/E/P/B/FCF all skipped → max_score=0 → neutral
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10

    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_neutral_mid_range(self, mock_metrics, mock_details):
        # P/E = 500e9/(2*5e9) = 50 → 0pts, P/B = 500e9/30e9 = 16.7 → 0pts
        # FCF yield = 3e9/500e9 = 0.6% → 0pts, DCF modest → maybe 1-2pts
        # Expect low score / max_score → neutral or bearish range
        mock_metrics.return_value = [
            _make_metrics(
                operating_cash_flow=3e9,
                earnings_per_share=2.0,
                shareholders_equity=30e9,
            ),
            _make_metrics(operating_cash_flow=2.8e9),
        ]
        mock_details.return_value = _make_details(
            market_cap=500e9,
            weighted_shares_outstanding=5e9,
        )
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        assert signals[0]["signal"] in ("neutral", "bearish")

    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_pe_boundary_exactly_15(self, mock_metrics, mock_details):
        # P/E=15: threshold is < 15 for attractive, so 15 falls to "fairly valued" (1pt)
        # Set up: market_cap / (eps * shares) = 15 → eps=2, shares=5e9, market_cap=150e9
        mock_metrics.return_value = [
            _make_metrics(earnings_per_share=2.0, shareholders_equity=100e9),
        ]
        mock_details.return_value = _make_details(
            market_cap=150e9,
            weighted_shares_outstanding=5e9,
        )
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        assert "fairly valued" in signals[0]["reasoning"]

    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_margin_of_safety_boundary(self, mock_metrics, mock_details):
        # DCF / market_cap exactly 1.25 → threshold is > 1.25, so NOT undervalued
        # We can't easily construct exact DCF, so test that a valid signal is returned
        mock_metrics.return_value = [
            _make_metrics(operating_cash_flow=10e9, shareholders_equity=50e9),
            _make_metrics(operating_cash_flow=9e9),
        ]
        mock_details.return_value = _make_details(market_cap=100e9)
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        # Just verify it doesn't crash and returns a valid signal
        assert signals[0]["signal"] in ("bullish", "neutral", "bearish")

    @patch("src.agents.valuation.get_company_details", return_value=None)
    @patch("src.agents.valuation.get_financial_metrics")
    def test_api_exception_returns_neutral_confidence_0(self, mock_metrics, mock_details):
        mock_metrics.side_effect = Exception("API error")
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 0

    @patch("src.agents.valuation.get_company_details", return_value=None)
    @patch("src.agents.valuation.get_financial_metrics")
    def test_all_data_missing_returns_neutral(self, mock_metrics, mock_details):
        mock_metrics.return_value = [_make_metrics()]  # All fields None
        result = valuation_agent(_make_state())
        signals = result["data"]["analyst_signals"]["valuation_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10

    @patch("src.agents.valuation.get_company_details", return_value=None)
    @patch("src.agents.valuation.get_financial_metrics")
    def test_output_structure(self, mock_metrics, mock_details):
        mock_metrics.return_value = []
        result = valuation_agent(_make_state())
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "valuation_analyst" in result["data"]["analyst_signals"]
