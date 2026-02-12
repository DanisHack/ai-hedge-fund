"""Tests for src/agents/pabrai.py — Mohnish Pabrai persona agent."""
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
        m.revenue = 15e9 * (1.05 ** (count - 1 - i))
        m.net_income = 2e9 * (1.07 ** (count - 1 - i))
        m.net_profit_margin = 0.14
        m.gross_profit_margin = 0.40
        m.return_on_equity = 0.14
        m.free_cash_flow = 2.5e9
        m.operating_cash_flow = 3.5e9
        m.debt_to_equity = 0.2
        m.earnings_per_share = 3.5
        m.shareholders_equity = 30e9
        m.current_ratio = 2.5
        m.total_assets = 50e9
        m.total_liabilities = 20e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 25e9
    details.weighted_shares_outstanding = 5e9
    details.share_class_shares_outstanding = 5e9
    details.total_employees = 15000
    return details


class TestPabraiAgent:
    @patch("src.agents.pabrai.call_llm")
    @patch("src.agents.pabrai.get_company_details")
    @patch("src.agents.pabrai.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.pabrai import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=80,
            reasoning="Asymmetric bet with low downside and potential double.",
        )

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.agent_id == "pabrai_analyst"

    @patch("src.agents.pabrai.call_llm")
    @patch("src.agents.pabrai.get_company_details")
    @patch("src.agents.pabrai.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.pabrai import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.pabrai.call_llm")
    @patch("src.agents.pabrai.get_company_details")
    @patch("src.agents.pabrai.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.pabrai import pabrai_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=70,
            reasoning="Dhandho setup — heads I win big.",
        )

        state = {
            "data": {"tickers": ["AAPL"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = pabrai_agent(state)

        assert "messages" in result
        signals = result["data"]["analyst_signals"]["pabrai_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_risk_reward_metrics(self):
        from src.agents.pabrai import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "Current Ratio" in facts
        assert "Debt/Equity" in facts
        assert "FCF Yield" in facts
        assert "P/E Ratio" in facts

    def test_facts_no_details(self):
        from src.agents.pabrai import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "Current Ratio" in facts
        assert "Market Cap" not in facts
