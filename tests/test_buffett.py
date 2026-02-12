"""Tests for src/agents/buffett.py — Warren Buffett persona agent."""
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
    """Build mock financial metrics with Buffett-relevant fields."""
    metrics = []
    for i in range(count):
        m = MagicMock()
        m.return_on_equity = 0.18 - i * 0.01
        m.debt_to_equity = 0.3
        m.net_profit_margin = 0.20 - i * 0.005
        m.net_income = 5e9 * (1.1 ** (count - 1 - i))
        m.revenue = 50e9 * (1.08 ** (count - 1 - i))
        m.operating_cash_flow = 8e9
        m.free_cash_flow = 6e9
        m.shareholders_equity = 100e9
        m.current_ratio = 1.8
        m.earnings_per_share = 6.0
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 200e9
    details.weighted_shares_outstanding = 15e9
    details.share_class_shares_outstanding = 15e9
    return details


class TestBuffettAgent:
    @patch("src.agents.buffett.call_llm")
    @patch("src.agents.buffett.get_company_details")
    @patch("src.agents.buffett.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.buffett import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=85,
            reasoning="Strong moat with margin of safety.",
        )

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.confidence == 85
        assert result.agent_id == "buffett_analyst"

    @patch("src.agents.buffett.call_llm")
    @patch("src.agents.buffett.get_company_details")
    @patch("src.agents.buffett.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.buffett import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL
        assert result.confidence == 0

    @patch("src.agents.buffett.call_llm")
    @patch("src.agents.buffett.get_company_details")
    @patch("src.agents.buffett.get_financial_metrics")
    def test_llm_failure_returns_neutral(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.buffett import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        # Simulate default_factory being used (LLM failed)
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="Unable to complete Buffett-style analysis.",
        )

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        assert result.signal == SignalType.NEUTRAL
        assert result.confidence == 0

    @patch("src.agents.buffett.call_llm")
    @patch("src.agents.buffett.get_company_details")
    @patch("src.agents.buffett.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.buffett import buffett_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=75,
            reasoning="Good business at fair price.",
        )

        state = {
            "data": {
                "tickers": ["AAPL"],
                "end_date": "2024-01-01",
            },
            "metadata": _make_metadata(),
        }

        result = buffett_agent(state)

        assert "messages" in result
        assert "data" in result
        signals = result["data"]["analyst_signals"]["buffett_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    @patch("src.agents.buffett.get_company_details")
    @patch("src.agents.buffett.get_financial_metrics")
    def test_facts_include_key_metrics(self, mock_metrics, mock_details):
        from src.agents.buffett import _build_facts

        metrics = _make_metrics()
        details = _make_details()

        facts = _build_facts(metrics, details)

        assert "ROE" in facts
        assert "Debt/Equity" in facts
        assert "Net Margin" in facts
        assert "Market Cap" in facts

    def test_facts_no_details(self):
        from src.agents.buffett import _build_facts

        metrics = _make_metrics(2)
        facts = _build_facts(metrics, None)

        assert "ROE" in facts
        assert "Market Cap" not in facts


class TestPersonaConfig:
    def test_buffett_in_persona_config(self):
        from src.config.agents import PERSONA_CONFIG
        assert "buffett" in PERSONA_CONFIG
        node_name, agent_func = PERSONA_CONFIG["buffett"]
        assert node_name == "buffett_analyst"
        assert callable(agent_func)

    def test_persona_config_separate_from_analysts(self):
        from src.config.agents import ANALYST_CONFIG, PERSONA_CONFIG
        analyst_keys = set(ANALYST_CONFIG.keys())
        persona_keys = set(PERSONA_CONFIG.keys())
        assert analyst_keys & persona_keys == set()


class TestWorkflowPersonas:
    def test_workflow_without_personas(self):
        from src.graph.workflow import create_workflow
        workflow = create_workflow()
        # Should have 6 analysts + risk_manager + portfolio_manager = 8 nodes
        assert len(workflow.nodes) == 8

    def test_workflow_with_buffett_persona(self):
        from src.graph.workflow import create_workflow
        workflow = create_workflow(personas=["buffett"])
        # Should have 6 analysts + 1 persona + risk_manager + portfolio_manager = 9 nodes
        assert len(workflow.nodes) == 9

    def test_workflow_with_unknown_persona_ignored(self):
        from src.graph.workflow import create_workflow
        workflow = create_workflow(personas=["nonexistent"])
        assert len(workflow.nodes) == 8

    def test_workflow_with_all_personas(self):
        from src.config.agents import PERSONA_CONFIG
        from src.graph.workflow import create_workflow
        workflow = create_workflow(personas=["all"])
        # 6 analysts + all personas + risk_manager + portfolio_manager
        assert len(workflow.nodes) == 6 + len(PERSONA_CONFIG) + 2
