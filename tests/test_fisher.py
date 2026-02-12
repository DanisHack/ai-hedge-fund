"""Tests for src/agents/fisher.py — Phil Fisher persona agent."""
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
        m.revenue = 40e9 * (1.12 ** (count - 1 - i))
        m.net_income = 6e9 * (1.14 ** (count - 1 - i))
        m.net_profit_margin = 0.16 + i * 0.005  # expanding margins (older = lower)
        m.gross_profit_margin = 0.55 + i * 0.005
        m.return_on_equity = 0.20 - i * 0.005
        m.free_cash_flow = 5e9
        m.operating_cash_flow = 8e9
        m.debt_to_equity = 0.3
        m.earnings_per_share = 6.0
        m.shareholders_equity = 70e9
        m.current_ratio = 1.8
        m.total_assets = 100e9
        m.total_liabilities = 30e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 180e9
    details.weighted_shares_outstanding = 10e9
    details.share_class_shares_outstanding = 10e9
    details.total_employees = 35000
    return details


class TestFisherAgent:
    @patch("src.agents.fisher.call_llm")
    @patch("src.agents.fisher.get_company_details")
    @patch("src.agents.fisher.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.fisher import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=80,
            reasoning="Outstanding growth with expanding margins and heavy reinvestment.",
        )

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.agent_id == "fisher_analyst"

    @patch("src.agents.fisher.call_llm")
    @patch("src.agents.fisher.get_company_details")
    @patch("src.agents.fisher.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.fisher import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.fisher.call_llm")
    @patch("src.agents.fisher.get_company_details")
    @patch("src.agents.fisher.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.fisher import fisher_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=75,
            reasoning="Growth company with scuttlebutt confirmation.",
        )

        state = {
            "data": {"tickers": ["AAPL"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = fisher_agent(state)

        assert "messages" in result
        signals = result["data"]["analyst_signals"]["fisher_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_growth_quality_metrics(self):
        from src.agents.fisher import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "Revenue" in facts
        assert "Gross Margin" in facts
        assert "ROE" in facts
        assert "Revenue Per Employee" in facts

    def test_facts_no_details(self):
        from src.agents.fisher import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "Revenue" in facts
        assert "Market Cap" not in facts
