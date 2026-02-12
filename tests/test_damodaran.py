"""Tests for src/agents/damodaran.py — Aswath Damodaran persona agent."""
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
        m.revenue = 60e9 * (1.08 ** (count - 1 - i))
        m.net_income = 8e9 * (1.10 ** (count - 1 - i))
        m.net_profit_margin = 0.18 - i * 0.005
        m.return_on_equity = 0.22 - i * 0.01
        m.free_cash_flow = 7e9
        m.operating_cash_flow = 10e9
        m.debt_to_equity = 0.4
        m.earnings_per_share = 7.0
        m.shareholders_equity = 90e9
        m.gross_profit_margin = 0.55
        m.current_ratio = 1.8
        m.total_assets = 150e9
        m.total_liabilities = 60e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 250e9
    details.weighted_shares_outstanding = 15e9
    details.share_class_shares_outstanding = 15e9
    details.total_employees = 50000
    return details


class TestDamodaranAgent:
    @patch("src.agents.damodaran.call_llm")
    @patch("src.agents.damodaran.get_company_details")
    @patch("src.agents.damodaran.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.damodaran import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=75,
            reasoning="Intrinsic value above market price with strong cash flows.",
        )

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.confidence == 75
        assert result.agent_id == "damodaran_analyst"

    @patch("src.agents.damodaran.call_llm")
    @patch("src.agents.damodaran.get_company_details")
    @patch("src.agents.damodaran.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.damodaran import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL
        assert result.confidence == 0

    @patch("src.agents.damodaran.call_llm")
    @patch("src.agents.damodaran.get_company_details")
    @patch("src.agents.damodaran.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.damodaran import damodaran_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=70,
            reasoning="Valuation supported by cash flows.",
        )

        state = {
            "data": {"tickers": ["AAPL"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = damodaran_agent(state)

        assert "messages" in result
        assert "data" in result
        signals = result["data"]["analyst_signals"]["damodaran_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_key_metrics(self):
        from src.agents.damodaran import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "Revenue" in facts
        assert "Free Cash Flow" in facts
        assert "ROE" in facts
        assert "DCF Estimate" in facts

    def test_facts_no_details(self):
        from src.agents.damodaran import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "Revenue" in facts
        assert "Market Cap" not in facts
