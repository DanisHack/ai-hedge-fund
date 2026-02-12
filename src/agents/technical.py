"""Technical analyst agent."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, LLMAnalysisResult, SignalType
from src.data.polygon_client import get_prices
from src.graph.state import AgentState
from src.llm import call_llm

logger = logging.getLogger(__name__)

AGENT_ID = "technical_analyst"


def technical_agent(state: AgentState) -> dict[str, Any]:
    """Analyze each ticker's price action and generate signals.

    Uses: SMA crossover (20/50), RSI(14), volume trend, price vs SMA50.
    """
    data = state["data"]
    tickers: list[str] = data.get("tickers", [])
    start_date: str = data.get("start_date", "")
    end_date: str = data.get("end_date", "")
    show_reasoning: bool = state["metadata"].get("show_reasoning", False)

    signals: list[dict] = []
    current_prices: dict[str, float] = {}

    for ticker in tickers:
        try:
            signal, latest_price = _analyze_ticker(ticker, start_date, end_date, metadata=state["metadata"])
            if latest_price is not None:
                current_prices[ticker] = latest_price
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
        "data": {
            "analyst_signals": {AGENT_ID: signals},
            "current_prices": current_prices,
        },
    }


def _analyze_ticker(
    ticker: str, start_date: str, end_date: str, metadata: dict | None = None,
) -> tuple[AnalystSignal, float | None]:
    """Run technical analysis on a single ticker. Returns (signal, latest_price)."""
    prices = get_prices(ticker, start_date, end_date)

    if len(prices) < 50:
        return (
            AnalystSignal(
                agent_id=AGENT_ID, ticker=ticker,
                signal=SignalType.NEUTRAL, confidence=10,
                reasoning=f"Insufficient price history ({len(prices)} bars, need 50+).",
            ),
            prices[-1].close if prices else None,
        )

    closes = np.array([p.close for p in prices])
    volumes = np.array([p.volume for p in prices])

    score = 0
    max_score = 0
    reasons: list[str] = []
    analysis_data: dict[str, Any] = {}

    # --- SMA Crossover (20 vs 50) ---
    sma_20 = float(np.mean(closes[-20:]))
    sma_50 = float(np.mean(closes[-50:]))
    analysis_data["sma_20"] = sma_20
    analysis_data["sma_50"] = sma_50
    max_score += 2
    if sma_20 > sma_50:
        score += 2
        pct_above = (sma_20 - sma_50) / sma_50 * 100
        reasons.append(f"SMA20 above SMA50 by {pct_above:.1f}% (bullish)")
    else:
        pct_below = (sma_50 - sma_20) / sma_50 * 100
        reasons.append(f"SMA20 below SMA50 by {pct_below:.1f}% (bearish)")

    # --- RSI(14) ---
    rsi = _compute_rsi(closes, period=14)
    analysis_data["rsi"] = rsi
    max_score += 2
    if rsi is not None:
        if rsi < 30:
            score += 2
            reasons.append(f"RSI={rsi:.1f} (oversold, potential reversal up)")
        elif rsi > 70:
            score += 0
            reasons.append(f"RSI={rsi:.1f} (overbought, potential pullback)")
        else:
            score += 1
            reasons.append(f"RSI={rsi:.1f} (neutral)")
    else:
        max_score -= 2

    # --- Volume Trend (10-day vs 50-day average) ---
    if len(volumes) >= 50:
        max_score += 1
        vol_10 = float(np.mean(volumes[-10:]))
        vol_50 = float(np.mean(volumes[-50:]))
        analysis_data["volume_ratio"] = vol_10 / vol_50 if vol_50 > 0 else 0
        if vol_50 > 0 and vol_10 > vol_50 * 1.2:
            score += 1
            reasons.append(f"Rising volume ({vol_10 / vol_50:.1f}x 50d avg)")
        else:
            ratio = vol_10 / vol_50 if vol_50 > 0 else 0
            reasons.append(f"Flat/declining volume ({ratio:.1f}x 50d avg)")

    # --- Price vs SMA50 ---
    max_score += 1
    current_price = float(closes[-1])
    analysis_data["current_price"] = current_price
    if current_price > sma_50:
        score += 1
        reasons.append(f"Price ${current_price:.2f} above SMA50 ${sma_50:.2f}")
    else:
        reasons.append(f"Price ${current_price:.2f} below SMA50 ${sma_50:.2f}")

    # --- Signal ---
    ratio = score / max_score if max_score > 0 else 0.5
    confidence = round(ratio * 100)

    if ratio >= 0.65:
        signal_type = SignalType.BULLISH
    elif ratio <= 0.35:
        signal_type = SignalType.BEARISH
    else:
        signal_type = SignalType.NEUTRAL

    rule_based = AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=signal_type, confidence=confidence,
        reasoning="; ".join(reasons),
    )

    if metadata and metadata.get("use_llm") and analysis_data:
        llm_signal = _llm_analyze(ticker, analysis_data, rule_based, metadata)
        return llm_signal, current_price

    return rule_based, current_price


def _llm_analyze(
    ticker: str,
    analysis_data: dict[str, Any],
    rule_based: AnalystSignal,
    metadata: dict,
) -> AnalystSignal:
    """Use LLM to reason about technical indicators."""
    facts = [
        f"- SMA20: ${analysis_data['sma_20']:.2f}",
        f"- SMA50: ${analysis_data['sma_50']:.2f}",
        f"- Current Price: ${analysis_data['current_price']:.2f}",
    ]
    if analysis_data.get("rsi") is not None:
        facts.append(f"- RSI(14): {analysis_data['rsi']:.1f}")
    if "volume_ratio" in analysis_data:
        facts.append(f"- Volume (10d/50d): {analysis_data['volume_ratio']:.1f}x")

    prompt = (
        f"You are a technical analyst evaluating {ticker}.\n\n"
        f"Technical Indicators:\n"
        + "\n".join(facts)
        + f"\n\nRule-based score: {rule_based.confidence}% ({rule_based.signal.value})\n\n"
        "Analyze the technical picture. Consider:\n"
        "1. SMA crossover: Is the short-term trend aligned with the long-term?\n"
        "2. RSI: Is the stock overbought/oversold? Any divergence?\n"
        "3. Volume: Does volume confirm the price trend?\n"
        "4. Price position: Where is price relative to key moving averages?\n"
        "5. Confluence: Do multiple indicators agree or conflict?\n\n"
        "Provide a trading signal (bullish/bearish/neutral), confidence 0-100, "
        "and 2-4 sentence reasoning citing specific data points."
    )

    result = call_llm(
        prompt=prompt,
        response_model=LLMAnalysisResult,
        model_name=metadata.get("model_name", "gpt-4o-mini"),
        model_provider=metadata.get("model_provider", "openai"),
        default_factory=lambda: LLMAnalysisResult(
            signal=rule_based.signal,
            confidence=rule_based.confidence,
            reasoning=rule_based.reasoning,
        ),
    )

    return AnalystSignal(
        agent_id=AGENT_ID, ticker=ticker,
        signal=result.signal, confidence=result.confidence,
        reasoning=result.reasoning,
    )


def _compute_rsi(closes: np.ndarray, period: int = 14) -> float | None:
    """Compute RSI for the most recent value."""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))
