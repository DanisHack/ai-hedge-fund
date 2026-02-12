"""Tests for fundamentals analyst agent."""
from unittest.mock import patch

from src.agents.fundamentals import fundamentals_agent
from src.data.models import FinancialMetrics


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


class TestFundamentalsAgent:
    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_strong_bullish_all_metrics_good(self, mock_metrics, mock_details):
        mock_metrics.return_value = [
            _make_metrics(net_profit_margin=0.20, return_on_equity=0.20, debt_to_equity=0.3, revenue=120e6),
            _make_metrics(revenue=100e6),
        ]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        assert signals[0]["signal"] == "bullish"
        assert signals[0]["confidence"] == 100  # 8/8

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_strong_bearish_all_metrics_bad(self, mock_metrics, mock_details):
        mock_metrics.return_value = [
            _make_metrics(net_profit_margin=0.02, return_on_equity=0.03, debt_to_equity=2.0, revenue=80e6),
            _make_metrics(revenue=100e6),
        ]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        assert signals[0]["signal"] == "bearish"
        assert signals[0]["confidence"] == 0  # 0/8

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_no_metrics_returns_neutral(self, mock_metrics, mock_details):
        mock_metrics.return_value = []
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_only_one_metric_no_revenue_growth(self, mock_metrics, mock_details):
        # With only 1 metric, revenue growth can't be computed (needs 2)
        mock_metrics.return_value = [
            _make_metrics(net_profit_margin=0.20, return_on_equity=0.20, debt_to_equity=0.3),
        ]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        # Score=6/6 (revenue growth excluded), ratio=1.0 → bullish
        assert signals[0]["signal"] == "bullish"

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_all_fields_none(self, mock_metrics, mock_details):
        # Metrics present but all optional fields are None → max_score=0
        mock_metrics.return_value = [_make_metrics()]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10
        assert "Insufficient data" in signals[0]["reasoning"]

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_moderate_metrics_neutral(self, mock_metrics, mock_details):
        # margin=0.08 (1/2), ROE=0.10 (1/2), D/E=1.0 (1/2), growth=5% (1/2) → 4/8=0.5
        mock_metrics.return_value = [
            _make_metrics(net_profit_margin=0.08, return_on_equity=0.10, debt_to_equity=1.0, revenue=105e6),
            _make_metrics(revenue=100e6),
        ]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 50

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_api_exception_returns_neutral_confidence_0(self, mock_metrics, mock_details):
        mock_metrics.side_effect = Exception("API error")
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 0

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_revenue_growth_zero_previous_revenue(self, mock_metrics, mock_details):
        # metrics[1].revenue = 0 → revenue growth check skipped (no division error)
        mock_metrics.return_value = [
            _make_metrics(net_profit_margin=0.20, revenue=100e6),
            _make_metrics(revenue=0),
        ]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        # Should not crash, revenue growth excluded from scoring
        assert signals[0]["signal"] in ("bullish", "neutral", "bearish")

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_boundary_margin_exactly_0_15(self, mock_metrics, mock_details):
        # margin=0.15: threshold is > 0.15 for strong, so this is "moderate" (1pt)
        mock_metrics.return_value = [_make_metrics(net_profit_margin=0.15)]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        # 1/2 = 0.5 → neutral
        assert signals[0]["confidence"] == 50

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_boundary_margin_exactly_0_05(self, mock_metrics, mock_details):
        # margin=0.05: threshold is > 0.05 for moderate, so this is "weak" (0pt)
        mock_metrics.return_value = [_make_metrics(net_profit_margin=0.05)]
        result = fundamentals_agent(_make_state())
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        # 0/2 = 0.0 → bearish
        assert signals[0]["signal"] == "bearish"
        assert signals[0]["confidence"] == 0

    @patch("src.agents.fundamentals.get_company_details", return_value=None)
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_multiple_tickers(self, mock_metrics, mock_details):
        mock_metrics.return_value = [_make_metrics(net_profit_margin=0.20)]
        result = fundamentals_agent(_make_state(tickers=("AAPL", "MSFT")))
        signals = result["data"]["analyst_signals"]["fundamentals_analyst"]
        assert len(signals) == 2
        assert {s["ticker"] for s in signals} == {"AAPL", "MSFT"}
