"""Tests for src/agents/burry.py — Michael Burry persona agent."""
from unittest.mock import MagicMock, patch

from src.data.models import LLMAnalysisResult, SignalType


def _make_metadata():
    return {
        "model_name": "gpt-4o-mini",
        "model_provider": "openai",
        "show_reasoning": False,
        "use_llm": True,
    }


def _make_metrics(count=4):
    metrics = []
    for i in range(count):
        m = MagicMock()
        m.revenue = 25e9
        m.net_income = 3e9
        m.net_profit_margin = 0.12
        m.gross_profit_margin = 0.35
        m.return_on_equity = 0.10
        m.free_cash_flow = 4e9
        m.operating_cash_flow = 6e9
        m.debt_to_equity = 0.5
        m.earnings_per_share = 4.0
        m.shareholders_equity = 50e9
        m.current_ratio = 2.2
        m.total_assets = 120e9
        m.total_liabilities = 70e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 40e9  # low market cap = potential deep value
    details.weighted_shares_outstanding = 8e9
    details.share_class_shares_outstanding = 8e9
    details.total_employees = 20000
    return details


class TestBurryAgent:
    @patch("src.agents.burry.call_llm")
    @patch("src.agents.burry.get_company_details")
    @patch("src.agents.burry.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.burry import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=85,
            reasoning="Deep value with strong asset backing and high FCF yield.",
        )

        result = _analyze_ticker("GM", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.agent_id == "burry_analyst"

    @patch("src.agents.burry.call_llm")
    @patch("src.agents.burry.get_company_details")
    @patch("src.agents.burry.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.burry import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("GM", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.burry.call_llm")
    @patch("src.agents.burry.get_company_details")
    @patch("src.agents.burry.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.burry import burry_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=75,
            reasoning="Contrarian deep value play.",
        )

        state = {
            "data": {"tickers": ["GM"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = burry_agent(state)

        assert "messages" in result
        signals = result["data"]["analyst_signals"]["burry_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_deep_value_metrics(self):
        from src.agents.burry import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "Net Asset Value" in facts
        assert "Price/Book" in facts
        assert "FCF Yield" in facts

    def test_facts_no_details(self):
        from src.agents.burry import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "Total Assets" in facts
        assert "Market Cap" not in facts
