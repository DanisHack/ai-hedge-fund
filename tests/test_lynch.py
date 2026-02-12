"""Tests for src/agents/lynch.py — Peter Lynch persona agent."""
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
        m.revenue = 10e9 * (1.15 ** (count - 1 - i))
        m.net_income = 1.5e9 * (1.18 ** (count - 1 - i))
        m.net_profit_margin = 0.15
        m.gross_profit_margin = 0.45
        m.return_on_equity = 0.18
        m.free_cash_flow = 1.8e9
        m.operating_cash_flow = 2.5e9
        m.debt_to_equity = 0.25
        m.earnings_per_share = 5.0 * (1.18 ** (count - 1 - i))
        m.shareholders_equity = 20e9
        m.current_ratio = 2.0
        m.total_assets = 35e9
        m.total_liabilities = 15e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 50e9
    details.weighted_shares_outstanding = 3e9
    details.share_class_shares_outstanding = 3e9
    details.total_employees = 8000
    return details


class TestLynchAgent:
    @patch("src.agents.lynch.call_llm")
    @patch("src.agents.lynch.get_company_details")
    @patch("src.agents.lynch.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.lynch import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=85,
            reasoning="Classic ten-bagger with PEG below 1.0.",
        )

        result = _analyze_ticker("SBUX", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.agent_id == "lynch_analyst"

    @patch("src.agents.lynch.call_llm")
    @patch("src.agents.lynch.get_company_details")
    @patch("src.agents.lynch.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.lynch import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("SBUX", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.lynch.call_llm")
    @patch("src.agents.lynch.get_company_details")
    @patch("src.agents.lynch.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.lynch import lynch_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=80,
            reasoning="Growth at reasonable price.",
        )

        state = {
            "data": {"tickers": ["SBUX"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = lynch_agent(state)

        assert "messages" in result
        signals = result["data"]["analyst_signals"]["lynch_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_peg_metrics(self):
        from src.agents.lynch import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "EPS" in facts
        assert "Revenue" in facts
        assert "Debt/Equity" in facts

    def test_facts_no_details(self):
        from src.agents.lynch import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "EPS" in facts
        assert "Market Cap" not in facts
