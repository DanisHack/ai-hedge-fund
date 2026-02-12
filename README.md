# AI Hedge Fund

A multi-agent AI hedge fund that analyzes stocks using 8 specialized agents, manages risk with correlation-aware position sizing and stop-loss protection, and trades via backtesting or live paper trading. Built with LangGraph and Polygon.io.

**For educational and research purposes only. Not financial advice.**

## Architecture

```
                        Market Data (Polygon.io)
                                │
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
        ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
        │ Fundamentals │ │  Technical  │ │  Sentiment  │
        │  Net margin  │ │ SMA, RSI,  │ │   News      │
        │  ROE, D/E   │ │ MACD, ADX, │ │  headline   │
        │  Rev growth  │ │ Bollinger  │ │  scoring    │
        └──────┬───────┘ └──────┬──────┘ └──────┬──────┘
               │                │               │
        ┌──────┴───────┐ ┌─────┴──────┐ ┌──────┴──────┐
        │  Valuation   │ │   Growth   │ │ Macro Regime│
        │  DCF, P/E,   │ │  Rev/EPS   │ │  Sector     │
        │  P/B, FCF    │ │  trends,   │ │  rotation,  │
        │  yield       │ │  margins   │ │  cyclicality│
        └──────┬───────┘ └─────┬──────┘ └──────┬──────┘
               │                │               │
               │  ┌─────────────┼─────────────┐ │
               │  │  Persona Agents (opt-in)  │ │
               │  │  ├── Warren Buffett       │ │
               │  │  └── Benjamin Graham      │ │
               │  └─────────────┬─────────────┘ │
               │                │               │
               └────────────────┼───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │         Risk Manager          │
                │  Consensus voting             │
                │  Volatility penalty           │
                │  Correlation group caps (40%) │
                │  Position limits (25%)        │
                │  Exposure caps (90%)          │
                └───────────────┬───────────────┘
                                ▼
                ┌───────────────────────────────┐
                │       Portfolio Manager       │
                │  Signal → buy/sell/hold       │
                │  Confidence-based sizing      │
                │  Stop-loss / take-profit      │
                └───────────────────────────────┘
```

All agents run in parallel via LangGraph fan-out, then converge through the risk manager and portfolio manager sequentially.

## Features

- **8 analyst agents** — fundamentals, technical (6 indicators), sentiment, valuation (DCF + relative), growth, macro regime, plus 2 opt-in investor personas (Buffett, Graham)
- **Dual mode** — rule-based scoring (no API key needed) or LLM-powered reasoning (`--use-llm`)
- **Risk management** — correlation-aware position sizing, volatility penalties, exposure caps, consensus voting
- **Downside protection** — fixed stop-loss, trailing stop-loss, and take-profit with automatic execution
- **Backtesting engine** — daily/weekly/monthly frequency, benchmark comparison (SPY), performance metrics, equity curve, JSON/CSV export
- **Paper trading** — persistent portfolio state, incremental trading cycles, full performance tracking
- **5 LLM providers** — OpenAI, Anthropic, Groq, Google, DeepSeek
- **292 unit tests** with CI on Python 3.11/3.12/3.13

## Quick Start

```bash
git clone https://github.com/DanisHack/ai-hedge-fund.git
cd ai-hedge-fund
pip install -e ".[dev]"
```

Create a `.env` file:

```bash
cp .env.example .env
```

**Required:** `POLYGON_API_KEY` — [free tier available](https://polygon.io)

**Optional (for `--use-llm` or `--personas`):** `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `DEEPSEEK_API_KEY`

## Usage

### Single Analysis

```bash
# Rule-based (no LLM key needed)
python -m src.main --ticker AAPL,MSFT,NVDA

# With LLM reasoning
python -m src.main --ticker AAPL --use-llm

# With investor persona
python -m src.main --ticker AAPL --personas buffett --use-llm

# All personas
python -m src.main --ticker AAPL --personas all --use-llm

# Different LLM provider
python -m src.main --ticker AAPL --use-llm --provider anthropic --model claude-sonnet-4-20250514
```

### Backtesting

```bash
# Basic backtest
python -m src.backtester \
  --ticker AAPL,MSFT,NVDA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# With stop-loss protection
python -m src.backtester \
  --ticker AAPL,MSFT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --stop-loss 0.10 \
  --trailing-stop 0.08 \
  --take-profit 0.20

# Daily frequency, custom commission, export results
python -m src.backtester \
  --ticker AAPL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --frequency daily \
  --commission 0.002 \
  --export results.json
```

**Backtest output:** summary panel, performance metrics (Sharpe, drawdown, Calmar), trade statistics (win rate, profit factor), trade log, ASCII equity curve.

### Paper Trading

```bash
# Start or continue a paper portfolio
python -m src.paper_trader run --ticker AAPL,MSFT,NVDA

# With stop-loss and LLM
python -m src.paper_trader run \
  --ticker AAPL,MSFT \
  --stop-loss 0.10 \
  --trailing-stop 0.08 \
  --use-llm

# Check portfolio status
python -m src.paper_trader status

# Reset portfolio
python -m src.paper_trader reset --ticker AAPL,MSFT --cash 200000
```

State is persisted to `paper_portfolio.json` between runs. Use `--state-file` to manage multiple portfolios.

## CLI Reference

### Common Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker / -t` | required | Comma-separated tickers |
| `--cash` | 100000 | Starting capital |
| `--use-llm` | false | Enable LLM reasoning |
| `--personas` | none | Investor personas (`buffett`, `graham`, or `all`) |
| `--model` | gpt-4o-mini | LLM model name |
| `--provider` | openai | LLM provider |
| `--show-reasoning` | false | Log agent reasoning |
| `--debug` | false | Debug logging |

### Backtester Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--start-date` | required | Start date (YYYY-MM-DD) |
| `--end-date` | required | End date (YYYY-MM-DD) |
| `--frequency / -f` | weekly | `daily`, `weekly`, or `monthly` |
| `--lookback` | 90 | Lookback window in days |
| `--benchmark` | SPY | Benchmark ticker (`none` to disable) |
| `--commission` | 0.001 | Commission rate (0.1%) |
| `--slippage` | 0.00005 | Slippage rate (0.005%) |
| `--stop-loss` | none | Fixed stop-loss (e.g. `0.10` = 10%) |
| `--trailing-stop` | none | Trailing stop from peak (e.g. `0.08` = 8%) |
| `--take-profit` | none | Take-profit target (e.g. `0.20` = 20%) |
| `--export` | none | Export to `.json` or `.csv` |

## Agents

### Core Analysts

| Agent | Signals From |
|-------|-------------|
| **Fundamentals** | Net profit margin, ROE, debt/equity, revenue growth |
| **Technical** | SMA crossover (20/50), RSI(14), MACD (12/26/9), Bollinger Bands (20/2), ADX(14), volume trend |
| **Sentiment** | News headline scoring (24 positive + 24 negative keyword categories) |
| **Valuation** | DCF (5yr, 10% WACC), P/E, P/B, FCF yield, 25% margin of safety |
| **Growth** | Revenue/earnings growth rates, acceleration, consistency, margin expansion |
| **Macro Regime** | Sector classification (11 sectors), cyclical vs defensive, sector momentum via ETFs |

### Persona Agents (opt-in, LLM-only)

| Persona | Style |
|---------|-------|
| **Warren Buffett** | Competitive moats, margin of safety, ROE consistency, management quality |
| **Benjamin Graham** | Graham Number, net-net value, P/E < 15, P/B < 1.5, earnings stability |

### Risk Manager

- **Consensus voting** — aggregates all analyst signals (bullish/bearish/neutral)
- **Volatility penalty** — reduces confidence for stocks with >3% daily volatility (20-day)
- **Correlation grouping** — BFS-based clustering of tickers with >0.7 correlation; groups capped at 40% of portfolio
- **Position limits** — max 25% per position, 90% total exposure
- **Exposure check** — penalizes new bullish signals when portfolio is near capacity

### Portfolio Manager

- Converts risk-adjusted signals to buy/sell/hold actions
- Confidence threshold: 50% minimum for execution
- Position sizing: `allocation = min(confidence/100, 1.0) * 50%`
- Minimum trade size: $100

## Stop-Loss & Take-Profit

Three types of automatic downside protection, checked before each trading cycle:

| Type | Trigger | Example |
|------|---------|---------|
| **Fixed stop-loss** | Price drops X% below avg cost | `--stop-loss 0.10` exits at 10% loss |
| **Trailing stop** | Price drops X% below peak (high water mark) | `--trailing-stop 0.08` exits at 8% from peak |
| **Take-profit** | Price rises X% above avg cost | `--take-profit 0.20` exits at 20% gain |

Priority: fixed stop > trailing stop > take-profit (first match wins per position).

## Project Structure

```
ai-hedge-fund/
├── src/
│   ├── agents/           # 6 core + 2 persona + risk manager + portfolio manager
│   ├── backtest/          # Engine, portfolio tracker, metrics, models, export
│   ├── config/            # Settings (env vars), agent registry
│   ├── data/              # Polygon.io client, data models, TTL cache
│   ├── graph/             # LangGraph workflow definition & state
│   ├── paper_trading/     # Runner, state persistence
│   ├── llm.py             # LLM factory (5 providers) + structured output caller
│   ├── main.py            # Single analysis CLI
│   ├── backtester.py      # Backtest CLI
│   └── paper_trader.py    # Paper trading CLI
├── tests/                 # 292 unit tests
├── .github/workflows/     # CI (Python 3.11/3.12/3.13)
├── pyproject.toml
└── .env.example
```

## LLM Providers

| Provider | Example Model | Env Variable |
|----------|--------------|--------------|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-sonnet-4-20250514` | `ANTHROPIC_API_KEY` |
| Groq | `llama3-8b-8192` | `GROQ_API_KEY` |
| Google | `gemini-pro` | `GOOGLE_API_KEY` |
| DeepSeek | `deepseek-chat` | `DEEPSEEK_API_KEY` |

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Lint
python -m ruff check src/ tests/
```

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment decisions
- No investment advice or guarantees provided
- Past performance does not indicate future results
- Consult a licensed financial advisor before making investment decisions

By using this software, you agree to use it solely for learning and research.

## License

MIT License — see [LICENSE](LICENSE) for details.
