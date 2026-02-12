"""Tests for portfolio manager agent."""
from src.agents.portfolio_manager import portfolio_manager_agent


def _make_state(risk_signals=None, current_prices=None, portfolio=None):
    if portfolio is None:
        portfolio = {"cash": 100_000, "positions": {}, "total_value": 100_000}
    return {
        "messages": [],
        "data": {
            "tickers": [s["ticker"] for s in (risk_signals or [])],
            "risk_adjusted_signals": risk_signals or [],
            "current_prices": current_prices or {},
            "portfolio": portfolio,
        },
        "metadata": {"show_reasoning": False},
    }


def _make_risk_signal(ticker, signal, confidence, max_position_size=25_000):
    return {
        "ticker": ticker, "signal": signal, "confidence": confidence,
        "reasoning": "test", "max_position_size": max_position_size,
    }


class TestPortfolioManagerAgent:
    def test_buy_bullish_high_confidence(self):
        signals = [_make_risk_signal("AAPL", "bullish", 80, max_position_size=25_000)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0},
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert len(positions) == 1
        assert positions[0]["action"] == "buy"
        assert positions[0]["quantity"] > 0

    def test_sell_bearish_high_confidence(self):
        signals = [_make_risk_signal("AAPL", "bearish", 70)]
        portfolio = {
            "cash": 50_000,
            "positions": {"AAPL": {"shares": 10, "avg_cost": 140.0}},
            "total_value": 51_400,
        }
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0}, portfolio=portfolio,
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "sell"
        assert positions[0]["quantity"] == 10

    def test_hold_below_buy_threshold(self):
        signals = [_make_risk_signal("AAPL", "bullish", 40)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0},
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "hold"
        assert positions[0]["quantity"] == 0

    def test_hold_below_sell_threshold(self):
        signals = [_make_risk_signal("AAPL", "bearish", 40)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0},
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "hold"

    def test_hold_neutral_signal(self):
        signals = [_make_risk_signal("AAPL", "neutral", 80)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0},
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "hold"

    def test_no_price_returns_hold(self):
        signals = [_make_risk_signal("AAPL", "bullish", 80)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={},
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "hold"
        assert "No price available" in positions[0]["reasoning"]

    def test_insufficient_cash_for_min_trade(self):
        signals = [_make_risk_signal("AAPL", "bullish", 50, max_position_size=150)]
        portfolio = {"cash": 50, "positions": {}, "total_value": 50}
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0}, portfolio=portfolio,
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "hold"
        assert "Insufficient cash" in positions[0]["reasoning"]

    def test_sell_no_existing_position(self):
        signals = [_make_risk_signal("AAPL", "bearish", 80)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0},
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "hold"
        assert "No position to sell" in positions[0]["reasoning"]

    def test_allocation_scales_with_confidence(self):
        # Higher confidence → more shares
        signals_low = [_make_risk_signal("AAPL", "bullish", 50, max_position_size=50_000)]
        result_low = portfolio_manager_agent(_make_state(
            risk_signals=signals_low, current_prices={"AAPL": 100.0},
        ))
        qty_low = result_low["data"]["portfolio_output"]["positions"][0]["quantity"]

        signals_high = [_make_risk_signal("AAPL", "bullish", 100, max_position_size=50_000)]
        result_high = portfolio_manager_agent(_make_state(
            risk_signals=signals_high, current_prices={"AAPL": 100.0},
        ))
        qty_high = result_high["data"]["portfolio_output"]["positions"][0]["quantity"]

        assert qty_high > qty_low

    def test_buy_limited_by_cash(self):
        signals = [_make_risk_signal("AAPL", "bullish", 100, max_position_size=100_000)]
        portfolio = {"cash": 200, "positions": {}, "total_value": 200}
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 50.0}, portfolio=portfolio,
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        assert positions[0]["action"] == "buy"
        # Max trade_value = min(100000 * 0.5, 200) = 200 → 4 shares at $50
        assert positions[0]["quantity"] == 4

    def test_buy_limited_by_max_position(self):
        signals = [_make_risk_signal("AAPL", "bullish", 100, max_position_size=500)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 50.0},
        ))
        positions = result["data"]["portfolio_output"]["positions"]
        # allocation = 1.0 * 0.5 = 0.5, trade_value = min(500 * 0.5, 100000) = 250 → 5 shares
        assert positions[0]["quantity"] == 5

    def test_cash_decremented_across_multiple_buys(self):
        signals = [
            _make_risk_signal("AAPL", "bullish", 80, max_position_size=50_000),
            _make_risk_signal("MSFT", "bullish", 80, max_position_size=50_000),
        ]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals,
            current_prices={"AAPL": 150.0, "MSFT": 400.0},
        ))
        output = result["data"]["portfolio_output"]
        # Both should buy, but total spent should not exceed initial cash
        total_spent = sum(
            p["quantity"] * {"AAPL": 150.0, "MSFT": 400.0}[p["ticker"]]
            for p in output["positions"] if p["action"] == "buy"
        )
        assert total_spent <= 100_000
        assert output["cash_remaining"] >= 0

    def test_output_structure(self):
        signals = [_make_risk_signal("AAPL", "bullish", 80)]
        result = portfolio_manager_agent(_make_state(
            risk_signals=signals, current_prices={"AAPL": 150.0},
        ))
        assert "messages" in result
        assert len(result["messages"]) == 1
        output = result["data"]["portfolio_output"]
        assert "positions" in output
        assert "cash_remaining" in output
        assert "total_value" in output
        pos = output["positions"][0]
        assert all(k in pos for k in ("ticker", "action", "quantity", "confidence", "reasoning"))
