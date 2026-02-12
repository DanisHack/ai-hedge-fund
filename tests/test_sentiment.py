"""Tests for sentiment analyst agent."""
from datetime import datetime
from unittest.mock import patch

from src.agents.sentiment import sentiment_agent
from src.data.models import CompanyNews


def _make_state(tickers=("AAPL",), start_date="2024-01-01", end_date="2024-06-01"):
    return {
        "messages": [],
        "data": {"tickers": list(tickers), "start_date": start_date, "end_date": end_date},
        "metadata": {"show_reasoning": False},
    }


def _make_article(title, description=None):
    return CompanyNews(
        title=title,
        published_utc=datetime(2024, 3, 1),
        article_url="https://example.com/news",
        description=description,
    )


class TestSentimentAgent:
    @patch("src.agents.sentiment.get_company_news")
    def test_bullish_majority_positive_news(self, mock_news):
        articles = [_make_article("Stock beats earnings, surge in revenue")] * 8 + [
            _make_article("Prices stable today"),
            _make_article("No major news"),
        ]
        mock_news.return_value = articles
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert signals[0]["signal"] == "bullish"
        assert signals[0]["confidence"] > 50

    @patch("src.agents.sentiment.get_company_news")
    def test_bearish_majority_negative_news(self, mock_news):
        articles = [_make_article("Stock misses estimates, decline expected")] * 8 + [
            _make_article("Prices stable today"),
            _make_article("No major news"),
        ]
        mock_news.return_value = articles
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert signals[0]["signal"] == "bearish"
        assert signals[0]["confidence"] > 50

    @patch("src.agents.sentiment.get_company_news")
    def test_neutral_mixed_news(self, mock_news):
        articles = (
            [_make_article("Stock beats estimates")] * 3
            + [_make_article("Stock misses targets")] * 3
            + [_make_article("Nothing happened today")] * 4
        )
        mock_news.return_value = articles
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert signals[0]["signal"] == "neutral"

    @patch("src.agents.sentiment.get_company_news")
    def test_no_news_returns_neutral_confidence_10(self, mock_news):
        mock_news.return_value = []
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 10
        assert "No recent news" in signals[0]["reasoning"]

    @patch("src.agents.sentiment.get_company_news")
    def test_low_sample_penalty(self, mock_news):
        # 3 positive articles (<5 threshold)
        articles = [_make_article("Stock beats record earnings")] * 3
        mock_news.return_value = articles
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        # Score=1.0, base confidence=min(90, 50+50)=90, minus 20=70
        assert signals[0]["confidence"] <= 70
        assert "Low sample size" in signals[0]["reasoning"]

    @patch("src.agents.sentiment.get_company_news")
    def test_boundary_score_exactly_0_2_is_neutral(self, mock_news):
        # 6 positive, 4 negative, 0 neutral -> score = (6-4)/10 = 0.2
        # threshold is > 0.2, so 0.2 should be NEUTRAL
        articles = (
            [_make_article("Stock beats earnings")] * 6
            + [_make_article("Stock misses target")] * 4
        )
        mock_news.return_value = articles
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert signals[0]["signal"] == "neutral"

    @patch("src.agents.sentiment.get_company_news")
    def test_confidence_capped_at_90(self, mock_news):
        # All 20 articles strongly positive -> score = 1.0
        # confidence = min(90, 50 + 1.0 * 50) = min(90, 100) = 90
        articles = [_make_article("Stock beats record surge growth")] * 20
        mock_news.return_value = articles
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert signals[0]["confidence"] == 90

    @patch("src.agents.sentiment.get_company_news")
    def test_api_exception_returns_neutral_confidence_0(self, mock_news):
        mock_news.side_effect = Exception("API error")
        result = sentiment_agent(_make_state())
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert signals[0]["signal"] == "neutral"
        assert signals[0]["confidence"] == 0
        assert "Analysis failed" in signals[0]["reasoning"]

    @patch("src.agents.sentiment.get_company_news")
    def test_multiple_tickers(self, mock_news):
        mock_news.return_value = [_make_article("Stock beats earnings")] * 10
        result = sentiment_agent(_make_state(tickers=("AAPL", "MSFT")))
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert len(signals) == 2
        tickers = {s["ticker"] for s in signals}
        assert tickers == {"AAPL", "MSFT"}

    @patch("src.agents.sentiment.get_company_news")
    def test_output_structure(self, mock_news):
        mock_news.return_value = []
        result = sentiment_agent(_make_state())
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "data" in result
        assert "analyst_signals" in result["data"]
        assert "sentiment_analyst" in result["data"]["analyst_signals"]
        signals = result["data"]["analyst_signals"]["sentiment_analyst"]
        assert isinstance(signals, list)
        signal = signals[0]
        assert all(k in signal for k in ("agent_id", "ticker", "signal", "confidence", "reasoning"))
