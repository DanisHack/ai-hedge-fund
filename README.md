# AI Hedge Fund

An AI-powered hedge fund system that uses multiple LLM agents to analyze US stocks and generate trading signals. Built with LangGraph for agent orchestration, Polygon.io for market data, and Alpaca for broker integration.

**This project is for educational and research purposes only. Not financial advice.**

## How It Works

```
Market Data (Polygon.io)
        |
        v
┌─────────────────────────────────┐
│   Analyst Agents (parallel)     │
│                                 │
│   Fundamental  |  Technical     │
│   Sentiment    |  Valuation     │
│   Growth       |  Macro         │
└────────────────┬────────────────┘
                 |
                 v
┌─────────────────────────────────┐
│       Risk Manager              │
│   Position sizing, volatility,  │
│   correlation, margin limits    │
└────────────────┬────────────────┘
                 |
                 v
┌─────────────────────────────────┐
│     Portfolio Manager           │
│   Final buy/sell/hold decisions │
└────────────────┬────────────────┘
                 |
                 v
        Alpaca (Paper/Live)
```

## Tech Stack

- **Agent Framework**: LangGraph (LangChain)
- **Market Data**: Polygon.io
- **Broker**: Alpaca (paper + live trading)
- **LLM Providers**: OpenAI, Anthropic, DeepSeek, Groq, Gemini, Ollama
- **Language**: Python 3.11+

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/DanisHack/ai-hedge-fund.git
cd ai-hedge-fund
poetry install
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

You need at minimum:
- One LLM API key (OpenAI, Anthropic, etc.)
- Polygon.io API key (free tier available)
- Alpaca API key (free, for paper trading)

### 3. Run

```bash
poetry run python src/main.py --ticker AAPL,MSFT,NVDA
```

### 4. Backtest

```bash
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-12-31
```

## Project Structure

```
ai-hedge-fund/
├── src/
│   ├── agents/        # LLM-powered analyst agents
│   ├── data/          # Polygon.io data layer
│   ├── broker/        # Alpaca broker integration
│   ├── risk/          # Risk management engine
│   ├── backtest/      # Backtesting engine
│   ├── graph/         # LangGraph workflow & state
│   ├── config/        # Settings & environment
│   └── utils/         # Helpers & display
├── tests/             # Test suite
├── app/               # Web UI (coming soon)
├── pyproject.toml
└── README.md
```

## Agents

| Agent | Strategy |
|-------|----------|
| Fundamentals | Financial metrics, quality scores, earnings analysis |
| Technical | Moving averages, RSI, MACD, volume analysis |
| Sentiment | News sentiment, insider trading patterns |
| Valuation | DCF, P/E, EV/EBITDA, relative valuation |
| Growth | Revenue & earnings growth trajectory |
| Risk Manager | Position sizing, volatility limits, correlation |
| Portfolio Manager | Final trading decisions, order generation |

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Past performance does not indicate future results
- Consult a licensed financial advisor for investment decisions

By using this software, you agree to use it solely for learning purposes.

## License

MIT License - see [LICENSE](LICENSE) for details.
