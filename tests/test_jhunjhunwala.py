"""Tests for src/agents/jhunjhunwala.py — Rakesh Jhunjhunwala persona agent."""
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
        m.revenue = 20e9 * (1.15 ** (count - 1 - i))
        m.net_income = 3e9 * (1.20 ** (count - 1 - i))
        m.net_profit_margin = 0.16 - i * 0.005
        m.gross_profit_margin = 0.45
        m.return_on_equity = 0.20 - i * 0.01
        m.free_cash_flow = 3e9
        m.operating_cash_flow = 4.5e9
        m.debt_to_equity = 0.4
        m.earnings_per_share = 4.0 * (1.20 ** (count - 1 - i))
        m.shareholders_equity = 40e9
        m.current_ratio = 1.8
        m.total_assets = 70e9
        m.total_liabilities = 30e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 80e9
    details.weighted_shares_outstanding = 8e9
    details.share_class_shares_outstanding = 8e9
    details.total_employees = 25000
    return details


class TestJhunjhunwalaAgent:
    @patch("src.agents.jhunjhunwala.call_llm")
    @patch("src.agents.jhunjhunwala.get_company_details")
    @patch("src.agents.jhunjhunwala.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.jhunjhunwala import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=85,
            reasoning="Market leader with accelerating earnings.",
        )

        result = _analyze_ticker("RELIANCE", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.agent_id == "jhunjhunwala_analyst"

    @patch("src.agents.jhunjhunwala.call_llm")
    @patch("src.agents.jhunjhunwala.get_company_details")
    @patch("src.agents.jhunjhunwala.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.jhunjhunwala import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("RELIANCE", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.jhunjhunwala.call_llm")
    @patch("src.agents.jhunjhunwala.get_company_details")
    @patch("src.agents.jhunjhunwala.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.jhunjhunwala import jhunjhunwala_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=80,
            reasoning="Growth with strong fundamentals.",
        )

        state = {
            "data": {"tickers": ["RELIANCE"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = jhunjhunwala_agent(state)

        assert "messages" in result
        signals = result["data"]["analyst_signals"]["jhunjhunwala_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_growth_metrics(self):
        from src.agents.jhunjhunwala import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "Earnings" in facts
        assert "ROE" in facts
        assert "Revenue" in facts

    def test_facts_no_details(self):
        from src.agents.jhunjhunwala import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "Earnings" in facts
        assert "Market Cap" not in facts
