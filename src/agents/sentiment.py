"""Sentiment analyst agent — news headlines + insider trading patterns."""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, SignalType
from src.data.polygon_client import get_company_news
from src.graph.state import AgentState

logger = logging.getLogger(__name__)

AGENT_ID = "sentiment_analyst"

# Keywords for simple sentiment classification
POSITIVE_KEYWORDS = [
    "beat", "beats", "exceeded", "surge", "surges", "record", "upgrade",
    "upgraded", "outperform", "growth", "profit", "gains", "bullish",
    "strong", "positive", "optimistic", "expansion", "partnership",
    "innovation", "breakthrough", "approval", "launch", "rally",
]
NEGATIVE_KEYWORDS = [
    "miss", "misses", "missed", "decline", "declines", "downgrade",
    "downgraded", "underperform", "loss", "losses", "bearish", "weak",
    "negative", "pessimistic", "layoff", "layoffs", "lawsuit", "recall",
    "investigation", "fine", "fined", "crash", "plunge", "warning",
    "debt", "bankruptcy", "fraud", "scandal",
]


def sentiment_agent(state: AgentState) -> dict[str, Any]:
    """Analyze news sentiment for each ticker.

    Scores news headlines using keyword matching and generates
    a bullish/bearish/neutral signal based on sentiment balance.
    """
    data = state["data"]
    tickers: list[str] = data.get("tickers", [])
    start_date: str = data.get("start_date", "")
    end_date: str = data.get("end_date", "")
    show_reasoning: bool = state["metadata"].get("show_reasoning", False)

    signals: list[dict] = []

    for ticker in tickers:
        try:
            signal = _analyze_ticker(ticker, start_date, end_date)
        except Exception as e:
            logger.warning(f"[{AGENT_ID}] Failed to analyze {ticker}: {e}")
            signal = AnalystSignal(
                agent_id=AGENT_ID, ticker=ticker,
                signal=SignalType.NEUTRAL, confidence=0,
                reasoning=f"Analysis failed: {e}",
            )

        signals.append(signal.model_dump(mode="json"))

        if show_reasoning:
            logger.info(f"[{AGENT_ID}] {ticker}: {signal.signal.value} "
                        f"(confidence={signal.confidence}) — {signal.reasoning}")

    message = HumanMessage(
        content=json.dumps({"agent": AGENT_ID, "signals": signals}),
        name=AGENT_ID,
    )
    return {
        "messages": [message],
        "data": {"analyst_signals": {AGENT_ID: signals}},
    }


def _analyze_ticker(ticker: str, start_date: str, end_date: str) -> AnalystSignal:
    """Run sentiment analysis on a single ticker."""
    news = get_company_news(ticker, start_date=start_date, end_date=end_date, limit=20)

    if not news:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="No recent news available.",
        )

    positive_count = 0
    negative_count = 0
    neutral_count = 0
    reasons: list[str] = []

    for article in news:
        text = (article.title + " " + (article.description or "")).lower()
        pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
        neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)

        if pos_hits > neg_hits:
            positive_count += 1
        elif neg_hits > pos_hits:
            negative_count += 1
        else:
            neutral_count += 1

    total = positive_count + negative_count + neutral_count
    reasons.append(f"{len(news)} articles analyzed: "
                   f"{positive_count} positive, {negative_count} negative, {neutral_count} neutral")

    # Calculate sentiment score (-1 to +1)
    if total > 0:
        sentiment_score = (positive_count - negative_count) / total
    else:
        sentiment_score = 0

    # Map to signal
    if sentiment_score > 0.2:
        signal = SignalType.BULLISH
        confidence = min(90, round(50 + sentiment_score * 50))
        reasons.append(f"Net positive sentiment ({sentiment_score:.2f})")
    elif sentiment_score < -0.2:
        signal = SignalType.BEARISH
        confidence = min(90, round(50 + abs(sentiment_score) * 50))
        reasons.append(f"Net negative sentiment ({sentiment_score:.2f})")
    else:
        signal = SignalType.NEUTRAL
        confidence = round(30 + (1 - abs(sentiment_score)) * 20)
        reasons.append(f"Mixed/neutral sentiment ({sentiment_score:.2f})")

    # Adjust confidence based on sample size
    if len(news) < 5:
        confidence = max(10, confidence - 20)
        reasons.append(f"Low sample size ({len(news)} articles), reduced confidence")

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=signal, confidence=confidence,
        reasoning="; ".join(reasons),
    )
