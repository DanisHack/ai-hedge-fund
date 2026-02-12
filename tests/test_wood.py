"""Tests for src/agents/wood.py — Cathie Wood persona agent."""
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
        m.revenue = 20e9 * (1.30 ** (count - 1 - i))  # high growth
        m.net_income = 1e9 * (1.40 ** (count - 1 - i))
        m.net_profit_margin = 0.08 + i * 0.005
        m.gross_profit_margin = 0.65 - i * 0.005
        m.return_on_equity = 0.15
        m.free_cash_flow = 2e9
        m.operating_cash_flow = 4e9
        m.debt_to_equity = 0.2
        m.earnings_per_share = 3.0
        m.shareholders_equity = 30e9
        m.current_ratio = 2.5
        m.total_assets = 50e9
        m.total_liabilities = 20e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 100e9
    details.weighted_shares_outstanding = 5e9
    details.share_class_shares_outstanding = 5e9
    details.total_employees = 10000
    return details


class TestWoodAgent:
    @patch("src.agents.wood.call_llm")
    @patch("src.agents.wood.get_company_details")
    @patch("src.agents.wood.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.wood import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=90,
            reasoning="Disruptive innovation with accelerating revenue.",
        )

        result = _analyze_ticker("TSLA", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.agent_id == "wood_analyst"

    @patch("src.agents.wood.call_llm")
    @patch("src.agents.wood.get_company_details")
    @patch("src.agents.wood.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.wood import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("TSLA", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.wood.call_llm")
    @patch("src.agents.wood.get_company_details")
    @patch("src.agents.wood.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.wood import wood_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=85,
            reasoning="Innovation platform with massive TAM.",
        )

        state = {
            "data": {"tickers": ["TSLA"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = wood_agent(state)

        assert "messages" in result
        signals = result["data"]["analyst_signals"]["wood_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_growth_metrics(self):
        from src.agents.wood import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "Revenue" in facts
        assert "Gross Margin" in facts
        assert "Price/Sales" in facts

    def test_facts_no_details(self):
        from src.agents.wood import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "Revenue" in facts
        assert "Market Cap" not in facts
