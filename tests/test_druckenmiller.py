"""Tests for src/agents/druckenmiller.py — Stanley Druckenmiller persona agent."""
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
        m.revenue = 50e9 * (1.12 ** (count - 1 - i))
        m.net_income = 7e9 * (1.15 ** (count - 1 - i))
        m.net_profit_margin = 0.16 - i * 0.005
        m.gross_profit_margin = 0.50
        m.return_on_equity = 0.20
        m.free_cash_flow = 6e9
        m.operating_cash_flow = 9e9
        m.debt_to_equity = 0.35
        m.earnings_per_share = 6.0 * (1.15 ** (count - 1 - i))
        m.shareholders_equity = 80e9
        m.current_ratio = 1.9
        m.total_assets = 130e9
        m.total_liabilities = 50e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 200e9
    details.weighted_shares_outstanding = 12e9
    details.share_class_shares_outstanding = 12e9
    details.total_employees = 45000
    return details


class TestDruckenmillerAgent:
    @patch("src.agents.druckenmiller.call_llm")
    @patch("src.agents.druckenmiller.get_company_details")
    @patch("src.agents.druckenmiller.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.druckenmiller import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=85,
            reasoning="Accelerating earnings with favorable macro backdrop.",
        )

        result = _analyze_ticker("NVDA", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.agent_id == "druckenmiller_analyst"

    @patch("src.agents.druckenmiller.call_llm")
    @patch("src.agents.druckenmiller.get_company_details")
    @patch("src.agents.druckenmiller.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.druckenmiller import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("NVDA", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL

    @patch("src.agents.druckenmiller.call_llm")
    @patch("src.agents.druckenmiller.get_company_details")
    @patch("src.agents.druckenmiller.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.druckenmiller import druckenmiller_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=90,
            reasoning="Asymmetric opportunity with macro tailwinds.",
        )

        state = {
            "data": {"tickers": ["NVDA"], "end_date": "2024-01-01"},
            "metadata": _make_metadata(),
        }

        result = druckenmiller_agent(state)

        assert "messages" in result
        signals = result["data"]["analyst_signals"]["druckenmiller_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    def test_facts_include_momentum_metrics(self):
        from src.agents.druckenmiller import _build_facts

        facts = _build_facts(_make_metrics(), _make_details())

        assert "Earnings" in facts
        assert "Revenue" in facts
        assert "Net Margin" in facts
        assert "Market Cap" in facts

    def test_facts_no_details(self):
        from src.agents.druckenmiller import _build_facts

        facts = _build_facts(_make_metrics(2), None)

        assert "Revenue" in facts
        assert "Market Cap" not in facts
