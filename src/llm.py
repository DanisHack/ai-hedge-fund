"""LLM utilities — model factory and structured-output caller."""
from __future__ import annotations

import logging
from typing import TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from src.config.settings import (
    ANTHROPIC_API_KEY,
    DEEPSEEK_API_KEY,
    GOOGLE_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    validate_llm_key,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def get_chat_model(
    model_name: str = "gpt-4o-mini",
    model_provider: str = "openai",
    temperature: float = 0.0,
) -> BaseChatModel:
    """Instantiate a LangChain chat model for the given provider."""
    validate_llm_key(model_provider)

    if model_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, api_key=OPENAI_API_KEY, temperature=temperature)
    elif model_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, api_key=ANTHROPIC_API_KEY, temperature=temperature)
    elif model_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=model_name, api_key=GROQ_API_KEY, temperature=temperature)
    elif model_provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name, google_api_key=GOOGLE_API_KEY, temperature=temperature,
        )
    elif model_provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name, api_key=DEEPSEEK_API_KEY,
            base_url="https://api.deepseek.com", temperature=temperature,
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: '{model_provider}'. "
            f"Supported: openai, anthropic, groq, google, deepseek"
        )


def call_llm(
    prompt: str,
    response_model: type[T],
    model_name: str = "gpt-4o-mini",
    model_provider: str = "openai",
    max_retries: int = 3,
    default_factory: callable | None = None,
) -> T:
    """Call an LLM with structured Pydantic output and retry logic.

    Falls back to default_factory() if all retries fail, or raises
    the last exception if no default_factory is provided.
    """
    llm = get_chat_model(model_name, model_provider)
    structured_llm = llm.with_structured_output(response_model)

    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = structured_llm.invoke(prompt)
            logger.debug(f"LLM call succeeded on attempt {attempt}")
            return result
        except Exception as e:
            last_exception = e
            logger.warning(f"LLM call attempt {attempt}/{max_retries} failed: {e}")

    logger.error(f"LLM call failed after {max_retries} attempts")
    if default_factory is not None:
        return default_factory()

    raise last_exception  # type: ignore[misc]
