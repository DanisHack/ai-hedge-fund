"""Tests for LLM mode in analyst agents.

Each agent class tests:
1. LLM called when use_llm=True and data exists
2. LLM result mapped to AnalystSignal correctly
3. No data → returns early neutral, LLM not called
"""
from unittest.mock import MagicMock, patch

from src.data.models import AnalystSignal, LLMAnalysisResult, SignalType


def _make_llm_result(signal=SignalType.BULLISH, confidence=80.0, reasoning="LLM says buy."):
    return LLMAnalysisResult(signal=signal, confidence=confidence, reasoning=reasoning)


def _make_metadata(use_llm=True):
    return {
        "model_name": "gpt-4o-mini",
        "model_provider": "openai",
        "show_reasoning": False,
        "use_llm": use_llm,
    }


# ── Fundamentals ───────────────────────────────────────────────────


class TestFundamentalsLlmMode:
    @patch("src.agents.fundamentals.call_llm")
    @patch("src.agents.fundamentals.get_company_details")
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_llm_called_when_enabled(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.fundamentals import _analyze_ticker

        metric = MagicMock()
        metric.net_profit_margin = 0.20
        metric.return_on_equity = 0.18
        metric.debt_to_equity = 0.4
        metric.revenue = 1_000_000
        metric2 = MagicMock()
        metric2.revenue = 800_000
        mock_metrics.return_value = [metric, metric2]
        mock_details.return_value = None
        mock_call_llm.return_value = _make_llm_result()

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.reasoning == "LLM says buy."

    @patch("src.agents.fundamentals.call_llm")
    @patch("src.agents.fundamentals.get_company_details")
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_no_data_returns_early_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.fundamentals import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL
        assert result.confidence == 10

    @patch("src.agents.fundamentals.call_llm")
    @patch("src.agents.fundamentals.get_company_details")
    @patch("src.agents.fundamentals.get_financial_metrics")
    def test_llm_not_called_when_disabled(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.fundamentals import _analyze_ticker

        metric = MagicMock()
        metric.net_profit_margin = 0.20
        metric.return_on_equity = 0.18
        metric.debt_to_equity = 0.4
        metric.revenue = 1_000_000
        metric2 = MagicMock()
        metric2.revenue = 800_000
        mock_metrics.return_value = [metric, metric2]
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata(use_llm=False))

        mock_call_llm.assert_not_called()
        assert isinstance(result, AnalystSignal)


# ── Sentiment ──────────────────────────────────────────────────────


class TestSentimentLlmMode:
    @patch("src.agents.sentiment.call_llm")
    @patch("src.agents.sentiment.get_company_news")
    def test_llm_called_when_enabled(self, mock_news, mock_call_llm):
        from src.agents.sentiment import _analyze_ticker

        article = MagicMock()
        article.title = "Company beats earnings expectations with record growth"
        article.description = "Strong Q4 results."
        mock_news.return_value = [article] * 10
        mock_call_llm.return_value = _make_llm_result(SignalType.BEARISH, 65.0, "LLM sees risk.")

        result = _analyze_ticker("AAPL", "2024-01-01", "2024-03-01", metadata=_make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BEARISH
        assert result.reasoning == "LLM sees risk."

    @patch("src.agents.sentiment.call_llm")
    @patch("src.agents.sentiment.get_company_news")
    def test_no_news_returns_early_no_llm(self, mock_news, mock_call_llm):
        from src.agents.sentiment import _analyze_ticker

        mock_news.return_value = []

        result = _analyze_ticker("AAPL", "2024-01-01", "2024-03-01", metadata=_make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.sentiment.call_llm")
    @patch("src.agents.sentiment.get_company_news")
    def test_llm_not_called_when_disabled(self, mock_news, mock_call_llm):
        from src.agents.sentiment import _analyze_ticker

        article = MagicMock()
        article.title = "Big gains"
        article.description = ""
        mock_news.return_value = [article] * 10

        result = _analyze_ticker("AAPL", "2024-01-01", "2024-03-01", metadata=_make_metadata(use_llm=False))

        mock_call_llm.assert_not_called()
        assert isinstance(result, AnalystSignal)


# ── Valuation ──────────────────────────────────────────────────────


class TestValuationLlmMode:
    @patch("src.agents.valuation.call_llm")
    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_llm_called_when_enabled(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.valuation import _analyze_ticker

        metric = MagicMock()
        metric.operating_cash_flow = 5e9
        metric.earnings_per_share = 6.0
        metric.shareholders_equity = 50e9
        metric.net_profit_margin = 0.25
        metric2 = MagicMock()
        metric2.operating_cash_flow = 4e9
        mock_metrics.return_value = [metric, metric2]

        details = MagicMock()
        details.market_cap = 100e9
        details.weighted_shares_outstanding = 10e9
        details.share_class_shares_outstanding = 10e9
        mock_details.return_value = details
        mock_call_llm.return_value = _make_llm_result(SignalType.NEUTRAL, 50.0, "Fairly valued.")

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.NEUTRAL
        assert result.reasoning == "Fairly valued."

    @patch("src.agents.valuation.call_llm")
    @patch("src.agents.valuation.get_company_details")
    @patch("src.agents.valuation.get_financial_metrics")
    def test_no_data_returns_early_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.valuation import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL


# ── Technical ──────────────────────────────────────────────────────


class TestTechnicalLlmMode:
    def _make_prices(self, n=60, base=100.0, trend=0.5):
        from datetime import datetime, timedelta
        from src.data.models import Price
        return [
            Price(
                open=base + i * trend,
                high=base + i * trend + 1,
                low=base + i * trend - 1,
                close=base + i * trend,
                volume=1_000_000 + i * 10_000,
                timestamp=datetime(2024, 1, 1) + timedelta(days=i),
            )
            for i in range(n)
        ]

    @patch("src.agents.technical.call_llm")
    @patch("src.agents.technical.get_prices")
    def test_llm_called_when_enabled(self, mock_prices, mock_call_llm):
        from src.agents.technical import _analyze_ticker

        mock_prices.return_value = self._make_prices(60)
        mock_call_llm.return_value = _make_llm_result(SignalType.BULLISH, 85.0, "Strong trend.")

        signal, price = _analyze_ticker("AAPL", "2024-01-01", "2024-03-01", metadata=_make_metadata())

        mock_call_llm.assert_called_once()
        assert signal.signal == SignalType.BULLISH
        assert signal.reasoning == "Strong trend."
        assert price is not None

    @patch("src.agents.technical.call_llm")
    @patch("src.agents.technical.get_prices")
    def test_insufficient_bars_no_llm(self, mock_prices, mock_call_llm):
        from src.agents.technical import _analyze_ticker

        mock_prices.return_value = self._make_prices(10)

        signal, price = _analyze_ticker("AAPL", "2024-01-01", "2024-03-01", metadata=_make_metadata())

        mock_call_llm.assert_not_called()
        assert signal.signal == SignalType.NEUTRAL

    @patch("src.agents.technical.call_llm")
    @patch("src.agents.technical.get_prices")
    def test_llm_not_called_when_disabled(self, mock_prices, mock_call_llm):
        from src.agents.technical import _analyze_ticker

        mock_prices.return_value = self._make_prices(60)

        signal, price = _analyze_ticker("AAPL", "2024-01-01", "2024-03-01", metadata=_make_metadata(use_llm=False))

        mock_call_llm.assert_not_called()
        assert isinstance(signal, AnalystSignal)


# ── Growth ─────────────────────────────────────────────────────────


class TestGrowthLlmMode:
    @patch("src.agents.growth.call_llm")
    @patch("src.agents.growth.get_financial_metrics")
    def test_llm_called_when_enabled(self, mock_metrics, mock_call_llm):
        from src.agents.growth import _analyze_ticker

        metrics = []
        for i in range(4):
            m = MagicMock()
            m.revenue = 1_000_000 * (1.2 ** (3 - i))
            m.net_income = 200_000 * (1.15 ** (3 - i))
            m.net_profit_margin = 0.20 + i * 0.01
            metrics.append(m)
        mock_metrics.return_value = metrics
        mock_call_llm.return_value = _make_llm_result(SignalType.BULLISH, 90.0, "Accelerating growth.")

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.reasoning == "Accelerating growth."

    @patch("src.agents.growth.call_llm")
    @patch("src.agents.growth.get_financial_metrics")
    def test_insufficient_data_no_llm(self, mock_metrics, mock_call_llm):
        from src.agents.growth import _analyze_ticker

        mock_metrics.return_value = [MagicMock()]

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.growth.call_llm")
    @patch("src.agents.growth.get_financial_metrics")
    def test_llm_not_called_when_disabled(self, mock_metrics, mock_call_llm):
        from src.agents.growth import _analyze_ticker

        metrics = []
        for i in range(4):
            m = MagicMock()
            m.revenue = 1_000_000 * (1.2 ** (3 - i))
            m.net_income = 200_000 * (1.15 ** (3 - i))
            m.net_profit_margin = 0.20 + i * 0.01
            metrics.append(m)
        mock_metrics.return_value = metrics

        result = _analyze_ticker("AAPL", "2024-01-01", metadata=_make_metadata(use_llm=False))

        mock_call_llm.assert_not_called()
        assert isinstance(result, AnalystSignal)
