"""Macro regime / market environment analyst agent."""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
from langchain_core.messages import HumanMessage

from src.data.models import AnalystSignal, LLMAnalysisResult, SignalType
from src.data.polygon_client import get_company_details, get_prices
from src.graph.state import AgentState
from src.llm import call_llm

logger = logging.getLogger(__name__)

AGENT_ID = "macro_regime_analyst"

# ── Sector ETF tickers ────────────────────────────────────────────────
SECTOR_ETFS: dict[str, str] = {
    "XLK": "technology",
    "XLF": "financials",
    "XLE": "energy",
    "XLV": "healthcare",
    "XLI": "industrials",
    "XLY": "consumer_discretionary",
    "XLP": "consumer_staples",
    "XLU": "utilities",
    "XLRE": "real_estate",
    "XLC": "communication",
    "XLB": "materials",
}

CYCLICAL_SECTORS = frozenset({
    "technology", "financials", "industrials",
    "consumer_discretionary", "materials", "energy",
})
DEFENSIVE_SECTORS = frozenset({
    "healthcare", "consumer_staples", "utilities",
    "real_estate", "communication",
})

# ── Ticker → sector lookup helpers ────────────────────────────────────

# SIC 2-digit prefix → sector
SIC_SECTOR_MAP: dict[str, str] = {
    "35": "technology", "36": "technology", "37": "technology",
    "38": "technology", "73": "technology",
    "48": "communication",
    "60": "financials", "61": "financials", "62": "financials",
    "63": "financials", "64": "financials", "67": "financials",
    "13": "energy", "29": "energy",
    "28": "healthcare", "80": "healthcare",
    "20": "consumer_staples", "21": "consumer_staples",
    "51": "consumer_staples", "54": "consumer_staples",
    "53": "consumer_discretionary", "56": "consumer_discretionary",
    "57": "consumer_discretionary", "58": "consumer_discretionary",
    "59": "consumer_discretionary", "70": "consumer_discretionary",
    "79": "consumer_discretionary",
    "34": "industrials", "33": "industrials", "32": "industrials",
    "15": "industrials", "16": "industrials", "17": "industrials",
    "40": "industrials", "42": "industrials", "44": "industrials",
    "45": "industrials",
    "49": "utilities",
    "65": "real_estate",
    "26": "materials", "24": "materials", "10": "materials",
}

# Common tickers with well-known sector assignments
TICKER_SECTOR_OVERRIDES: dict[str, str] = {
    "AAPL": "technology", "MSFT": "technology", "GOOGL": "technology",
    "GOOG": "technology", "NVDA": "technology", "AMD": "technology",
    "INTC": "technology", "CRM": "technology", "ORCL": "technology",
    "META": "communication", "NFLX": "communication", "DIS": "communication",
    "AMZN": "consumer_discretionary", "TSLA": "consumer_discretionary",
    "HD": "consumer_discretionary", "NKE": "consumer_discretionary",
    "JPM": "financials", "BAC": "financials", "GS": "financials",
    "MS": "financials", "V": "financials", "MA": "financials",
    "XOM": "energy", "CVX": "energy", "COP": "energy",
    "JNJ": "healthcare", "UNH": "healthcare", "PFE": "healthcare",
    "LLY": "healthcare", "ABBV": "healthcare",
    "PG": "consumer_staples", "KO": "consumer_staples",
    "PEP": "consumer_staples", "WMT": "consumer_staples",
    "CAT": "industrials", "BA": "industrials", "UPS": "industrials",
    "HON": "industrials", "GE": "industrials",
    "NEE": "utilities", "DUK": "utilities",
}


# ── Market Regime data class ──────────────────────────────────────────


class MarketRegime:
    """Computed market-level regime indicators."""

    def __init__(self) -> None:
        self.spy_volatility: float | None = None
        self.spy_above_sma50: bool | None = None
        self.spy_above_sma200: bool | None = None
        self.leading_sectors: list[str] = []
        self.lagging_sectors: list[str] = []
        self.sector_returns: dict[str, float] = {}
        self.cyclical_vs_defensive: float = 0.0
        self.breadth_pct: float | None = None
        self.score: int = 0
        self.max_score: int = 0
        self.reasons: list[str] = []


# ── Agent entry point ─────────────────────────────────────────────────


def macro_regime_agent(state: AgentState) -> dict[str, Any]:
    """Assess market regime and generate per-ticker signals.

    Analyzes: SPY realized volatility, sector rotation, market breadth,
    and SPY trend to determine the current market environment.
    """
    data = state["data"]
    tickers: list[str] = data.get("tickers", [])
    start_date: str = data.get("start_date", "")
    end_date: str = data.get("end_date", "")
    show_reasoning: bool = state["metadata"].get("show_reasoning", False)

    try:
        regime = _compute_regime(start_date, end_date, tickers)
    except Exception as e:
        logger.warning(f"[{AGENT_ID}] Failed to compute market regime: {e}")
        regime = None

    signals: list[dict] = []

    for ticker in tickers:
        try:
            signal = _analyze_ticker(
                ticker, regime, start_date, end_date,
                metadata=state["metadata"],
            )
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


# ── Regime computation ────────────────────────────────────────────────


def _compute_regime(
    start_date: str,
    end_date: str,
    tickers: list[str],
) -> MarketRegime:
    """Compute market-level regime indicators from Polygon data."""
    regime = MarketRegime()

    # 1. SPY volatility + trend
    spy_prices = get_prices("SPY", start_date, end_date)
    if len(spy_prices) >= 50:
        spy_closes = np.array([p.close for p in spy_prices])

        # 20-day realized volatility, annualized
        returns_20d = np.diff(spy_closes[-21:]) / spy_closes[-21:-1]
        daily_vol = float(np.std(returns_20d, ddof=1))
        regime.spy_volatility = daily_vol * np.sqrt(252) * 100

        regime.max_score += 2
        if regime.spy_volatility < 15:
            regime.score += 2
            regime.reasons.append(
                f"Low SPY volatility ({regime.spy_volatility:.1f}% annualized, risk-on)")
        elif regime.spy_volatility > 25:
            regime.reasons.append(
                f"High SPY volatility ({regime.spy_volatility:.1f}% annualized, risk-off)")
        else:
            regime.score += 1
            regime.reasons.append(
                f"Moderate SPY volatility ({regime.spy_volatility:.1f}% annualized)")

        # SPY trend
        regime.max_score += 2
        sma_50 = float(np.mean(spy_closes[-50:]))
        latest_spy = float(spy_closes[-1])
        regime.spy_above_sma50 = latest_spy > sma_50

        if len(spy_closes) >= 200:
            sma_200 = float(np.mean(spy_closes[-200:]))
            regime.spy_above_sma200 = latest_spy > sma_200
        else:
            sma_200 = None

        if regime.spy_above_sma50 and regime.spy_above_sma200:
            regime.score += 2
            regime.reasons.append(
                "SPY above SMA50 and SMA200 (bullish trend)")
        elif regime.spy_above_sma50:
            regime.score += 1
            regime.reasons.append("SPY above SMA50 (mixed trend)")
        else:
            regime.reasons.append("SPY below SMA50 (bearish trend)")

    # 2. Sector rotation
    sector_returns: dict[str, float] = {}
    for etf, sector in SECTOR_ETFS.items():
        try:
            etf_prices = get_prices(etf, start_date, end_date)
            if len(etf_prices) >= 20:
                closes = np.array([p.close for p in etf_prices])
                ret_20d = (closes[-1] - closes[-20]) / closes[-20]
                sector_returns[sector] = float(ret_20d)
        except Exception:
            continue

    regime.sector_returns = sector_returns

    if len(sector_returns) >= 6:
        regime.max_score += 2
        sorted_sectors = sorted(sector_returns.items(), key=lambda x: x[1], reverse=True)
        regime.leading_sectors = [s for s, _ in sorted_sectors[:3]]
        regime.lagging_sectors = [s for s, _ in sorted_sectors[-3:]]

        cyclical_rets = [r for s, r in sector_returns.items() if s in CYCLICAL_SECTORS]
        defensive_rets = [r for s, r in sector_returns.items() if s in DEFENSIVE_SECTORS]
        cyclical_avg = float(np.mean(cyclical_rets)) if cyclical_rets else 0.0
        defensive_avg = float(np.mean(defensive_rets)) if defensive_rets else 0.0
        regime.cyclical_vs_defensive = cyclical_avg - defensive_avg

        if regime.cyclical_vs_defensive > 0.01:
            regime.score += 2
            regime.reasons.append(
                f"Cyclical sectors leading defensives by {regime.cyclical_vs_defensive:.1%} (bullish rotation)")
        elif regime.cyclical_vs_defensive < -0.01:
            regime.reasons.append(
                f"Defensive sectors leading cyclicals by {abs(regime.cyclical_vs_defensive):.1%} (bearish rotation)")
        else:
            regime.score += 1
            regime.reasons.append("Sector rotation neutral")

    # 3. Market breadth
    above_sma_count = 0
    total_count = 0
    for ticker in tickers:
        try:
            t_prices = get_prices(ticker, start_date, end_date)
            if len(t_prices) >= 50:
                t_closes = np.array([p.close for p in t_prices])
                t_sma50 = float(np.mean(t_closes[-50:]))
                total_count += 1
                if t_closes[-1] > t_sma50:
                    above_sma_count += 1
        except Exception:
            continue

    if total_count > 0:
        regime.breadth_pct = above_sma_count / total_count
        regime.max_score += 2
        if regime.breadth_pct >= 0.7:
            regime.score += 2
            regime.reasons.append(
                f"Strong breadth: {above_sma_count}/{total_count} tickers above 50d SMA")
        elif regime.breadth_pct >= 0.4:
            regime.score += 1
            regime.reasons.append(
                f"Mixed breadth: {above_sma_count}/{total_count} tickers above 50d SMA")
        else:
            regime.reasons.append(
                f"Weak breadth: {above_sma_count}/{total_count} tickers above 50d SMA")

    return regime


# ── Per-ticker analysis ───────────────────────────────────────────────


def _get_ticker_sector(ticker: str) -> str | None:
    """Determine a ticker's sector via overrides, then SIC code fallback."""
    if ticker in TICKER_SECTOR_OVERRIDES:
        return TICKER_SECTOR_OVERRIDES[ticker]

    try:
        details = get_company_details(ticker)
        if details and details.sic_code:
            return SIC_SECTOR_MAP.get(details.sic_code[:2])
    except Exception:
        pass
    return None


def _analyze_ticker(
    ticker: str,
    regime: MarketRegime | None,
    start_date: str,
    end_date: str,
    metadata: dict | None = None,
) -> AnalystSignal:
    """Apply market regime to a specific ticker, modulated by sector alignment."""
    if regime is None or regime.max_score == 0:
        return AnalystSignal(
            agent_id=AGENT_ID, ticker=ticker,
            signal=SignalType.NEUTRAL, confidence=10,
            reasoning="Unable to determine market regime (insufficient data).",
        )

    score = regime.score
    max_score = regime.max_score
    reasons = list(regime.reasons)

    # Sector alignment bonus
    ticker_sector = _get_ticker_sector(ticker)
    if ticker_sector and regime.sector_returns:
        max_score += 2
        if ticker_sector in regime.leading_sectors:
            score += 2
            reasons.append(f"{ticker} in leading sector ({ticker_sector})")
        elif ticker_sector in regime.lagging_sectors:
            reasons.append(f"{ticker} in lagging sector ({ticker_sector})")
        else:
            score += 1
            reasons.append(f"{ticker} in mid-performing sector ({ticker_sector})")

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

    if metadata and metadata.get("use_llm") and reasons:
        return _llm_analyze(ticker, regime, ticker_sector, rule_based, metadata)

    return rule_based


def _llm_analyze(
    ticker: str,
    regime: MarketRegime,
    ticker_sector: str | None,
    rule_based: AnalystSignal,
    metadata: dict,
) -> AnalystSignal:
    """Use LLM to reason about market regime impact on ticker."""
    facts: list[str] = []
    if regime.spy_volatility is not None:
        facts.append(f"- SPY Realized Volatility: {regime.spy_volatility:.1f}% annualized")
    if regime.spy_above_sma50 is not None:
        facts.append(f"- SPY vs SMA50: {'above' if regime.spy_above_sma50 else 'below'}")
    if regime.spy_above_sma200 is not None:
        facts.append(f"- SPY vs SMA200: {'above' if regime.spy_above_sma200 else 'below'}")
    if regime.leading_sectors:
        facts.append(f"- Leading Sectors (20d): {', '.join(regime.leading_sectors)}")
        facts.append(f"- Lagging Sectors (20d): {', '.join(regime.lagging_sectors)}")
        facts.append(f"- Cyclical vs Defensive: {regime.cyclical_vs_defensive:+.1%}")
    if regime.breadth_pct is not None:
        facts.append(f"- Market Breadth: {regime.breadth_pct:.0%} above 50d SMA")
    if ticker_sector:
        facts.append(f"- {ticker} Sector: {ticker_sector}")

    prompt = (
        f"You are a macro/market regime analyst evaluating how the current market "
        f"environment affects {ticker}.\n\n"
        f"Market Regime Indicators:\n"
        + "\n".join(facts)
        + f"\n\nRule-based regime score: {rule_based.confidence}% ({rule_based.signal.value})\n\n"
        "Analyze the market regime. Consider:\n"
        "1. Volatility: Is the market in risk-on or risk-off mode?\n"
        "2. Trend: Is SPY in an uptrend or downtrend (SMA positioning)?\n"
        "3. Sector rotation: Are cyclical or defensive sectors leading?\n"
        "4. Breadth: Is the rally/decline broad-based or narrow?\n"
        "5. Ticker alignment: Does this stock's sector benefit from the current regime?\n\n"
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
