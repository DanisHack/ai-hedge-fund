"""Tests for src/llm.py — model factory and call_llm."""
from unittest.mock import MagicMock, patch

import pytest

from src.data.models import LLMAnalysisResult, SignalType
from src.llm import call_llm, get_chat_model


class TestGetChatModel:
    @patch("src.llm.validate_llm_key")
    @patch("src.llm.OPENAI_API_KEY", "test-key")
    def test_openai_provider(self, mock_validate):
        model = get_chat_model("gpt-4o-mini", "openai")
        from langchain_openai import ChatOpenAI
        assert isinstance(model, ChatOpenAI)

    @patch("src.llm.validate_llm_key")
    @patch("src.llm.GROQ_API_KEY", "test-key")
    def test_groq_provider(self, mock_validate):
        model = get_chat_model("llama3-8b-8192", "groq")
        from langchain_groq import ChatGroq
        assert isinstance(model, ChatGroq)

    @patch("src.llm.validate_llm_key")
    @patch("src.llm.DEEPSEEK_API_KEY", "test-key")
    def test_deepseek_uses_openai_compatible(self, mock_validate):
        model = get_chat_model("deepseek-chat", "deepseek")
        from langchain_openai import ChatOpenAI
        assert isinstance(model, ChatOpenAI)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_chat_model("some-model", "unknown_provider")

    def test_missing_api_key_raises(self):
        with patch("src.config.settings.OPENAI_API_KEY", ""):
            with pytest.raises(ValueError):
                get_chat_model("gpt-4o-mini", "openai")


class TestCallLlm:
    def _mock_llm(self, invoke_side_effect):
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = invoke_side_effect
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        return mock_llm, mock_structured

    def test_successful_call(self):
        result_obj = LLMAnalysisResult(
            signal=SignalType.BULLISH, confidence=75.0, reasoning="Strong fundamentals.",
        )
        mock_llm, mock_structured = self._mock_llm([result_obj])

        with patch("src.llm.get_chat_model", return_value=mock_llm):
            result = call_llm("Analyze AAPL", LLMAnalysisResult)
            assert result.signal == SignalType.BULLISH
            assert result.confidence == 75.0
            mock_structured.invoke.assert_called_once_with("Analyze AAPL")

    def test_retry_then_succeed(self):
        result_obj = LLMAnalysisResult(
            signal=SignalType.NEUTRAL, confidence=50.0, reasoning="Mixed.",
        )
        mock_llm, mock_structured = self._mock_llm([Exception("timeout"), result_obj])

        with patch("src.llm.get_chat_model", return_value=mock_llm):
            result = call_llm("Analyze MSFT", LLMAnalysisResult, max_retries=3)
            assert result.signal == SignalType.NEUTRAL
            assert mock_structured.invoke.call_count == 2

    def test_all_retries_fail_uses_default(self):
        mock_llm, mock_structured = self._mock_llm(Exception("fail"))
        default = LLMAnalysisResult(
            signal=SignalType.BEARISH, confidence=30.0, reasoning="Fallback.",
        )

        with patch("src.llm.get_chat_model", return_value=mock_llm):
            result = call_llm(
                "Analyze TSLA", LLMAnalysisResult, max_retries=3,
                default_factory=lambda: default,
            )
            assert result.signal == SignalType.BEARISH
            assert mock_structured.invoke.call_count == 3

    def test_all_retries_fail_no_default_raises(self):
        mock_llm, mock_structured = self._mock_llm(Exception("fail"))

        with patch("src.llm.get_chat_model", return_value=mock_llm):
            with pytest.raises(Exception, match="fail"):
                call_llm("Analyze TSLA", LLMAnalysisResult, max_retries=2)
