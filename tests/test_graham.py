"""Tests for src/agents/graham.py — Benjamin Graham persona agent."""
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
    """Build mock financial metrics with Graham-relevant fields."""
    metrics = []
    for i in range(count):
        m = MagicMock()
        m.earnings_per_share = 5.0 - i * 0.2
        m.net_income = 3e9 * (1.05 ** (count - 1 - i))
        m.revenue = 40e9 * (1.06 ** (count - 1 - i))
        m.shareholders_equity = 80e9
        m.total_assets = 150e9
        m.total_liabilities = 60e9
        m.debt_to_equity = 0.4
        m.current_ratio = 2.2
        m.net_profit_margin = 0.15 - i * 0.005
        m.return_on_equity = 0.12
        m.operating_cash_flow = 5e9
        m.free_cash_flow = 3e9
        metrics.append(m)
    return metrics


def _make_details():
    details = MagicMock()
    details.market_cap = 120e9
    details.weighted_shares_outstanding = 10e9
    details.share_class_shares_outstanding = 10e9
    return details


class TestGrahamAgent:
    @patch("src.agents.graham.call_llm")
    @patch("src.agents.graham.get_company_details")
    @patch("src.agents.graham.get_financial_metrics")
    def test_llm_called_with_data(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.graham import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=80,
            reasoning="Strong margin of safety with Graham Number well above price.",
        )

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_called_once()
        assert result.signal == SignalType.BULLISH
        assert result.confidence == 80
        assert result.agent_id == "graham_analyst"

    @patch("src.agents.graham.call_llm")
    @patch("src.agents.graham.get_company_details")
    @patch("src.agents.graham.get_financial_metrics")
    def test_no_data_returns_neutral_no_llm(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.graham import _analyze_ticker

        mock_metrics.return_value = []
        mock_details.return_value = None

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        mock_call_llm.assert_not_called()
        assert result.signal == SignalType.NEUTRAL
        assert result.confidence == 0

    @patch("src.agents.graham.call_llm")
    @patch("src.agents.graham.get_company_details")
    @patch("src.agents.graham.get_financial_metrics")
    def test_llm_failure_returns_neutral(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.graham import _analyze_ticker

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.NEUTRAL, confidence=0,
            reasoning="Unable to complete Graham-style analysis.",
        )

        result = _analyze_ticker("AAPL", "2024-01-01", _make_metadata())

        assert result.signal == SignalType.NEUTRAL
        assert result.confidence == 0

    @patch("src.agents.graham.call_llm")
    @patch("src.agents.graham.get_company_details")
    @patch("src.agents.graham.get_financial_metrics")
    def test_outer_agent_structure(self, mock_metrics, mock_details, mock_call_llm):
        from src.agents.graham import graham_agent

        mock_metrics.return_value = _make_metrics()
        mock_details.return_value = _make_details()
        mock_call_llm.return_value = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=70,
            reasoning="Adequate margin of safety with stable earnings.",
        )

        state = {
            "data": {
                "tickers": ["AAPL"],
                "end_date": "2024-01-01",
            },
            "metadata": _make_metadata(),
        }

        result = graham_agent(state)

        assert "messages" in result
        assert "data" in result
        signals = result["data"]["analyst_signals"]["graham_analyst"]
        assert len(signals) == 1
        assert signals[0]["signal"] == "bullish"


class TestBuildFacts:
    @patch("src.agents.graham.get_company_details")
    @patch("src.agents.graham.get_financial_metrics")
    def test_facts_include_graham_metrics(self, mock_metrics, mock_details):
        from src.agents.graham import _build_facts

        metrics = _make_metrics()
        details = _make_details()

        facts = _build_facts(metrics, details)

        assert "EPS" in facts
        assert "Debt/Equity" in facts
        assert "Graham Number" in facts
        assert "Book Value Per Share" in facts

    def test_facts_no_details(self):
        from src.agents.graham import _build_facts

        metrics = _make_metrics(2)
        facts = _build_facts(metrics, None)

        assert "EPS" in facts
        assert "Graham Number" not in facts  # needs BVPS from details
        assert "Market Cap" not in facts

    def test_facts_with_margin_of_safety(self):
        from src.agents.graham import _build_facts

        metrics = _make_metrics()
        details = _make_details()

        facts = _build_facts(metrics, details)

        assert "Margin of Safety" in facts

    def test_facts_pe_and_pb_ratios(self):
        from src.agents.graham import _build_facts

        metrics = _make_metrics()
        details = _make_details()

        facts = _build_facts(metrics, details)

        assert "P/E Ratio" in facts
        assert "P/B Ratio" in facts


class TestPersonaConfig:
    def test_graham_in_persona_config(self):
        from src.config.agents import PERSONA_CONFIG
        assert "graham" in PERSONA_CONFIG
        node_name, agent_func = PERSONA_CONFIG["graham"]
        assert node_name == "graham_analyst"
        assert callable(agent_func)


class TestWorkflowPersonas:
    def test_workflow_with_graham_persona(self):
        from src.graph.workflow import create_workflow
        workflow = create_workflow(personas=["graham"])
        # 6 analysts + 1 persona + risk_manager + portfolio_manager = 9
        assert len(workflow.nodes) == 9

    def test_workflow_with_both_personas(self):
        from src.graph.workflow import create_workflow
        workflow = create_workflow(personas=["buffett", "graham"])
        # 6 analysts + 2 personas + risk_manager + portfolio_manager = 10
        assert len(workflow.nodes) == 10

    def test_workflow_with_all_includes_graham(self):
        from src.graph.workflow import create_workflow
        workflow = create_workflow(personas=["all"])
        # 6 analysts + 2 personas (buffett + graham) + risk_manager + portfolio_manager = 10
        assert len(workflow.nodes) == 10
