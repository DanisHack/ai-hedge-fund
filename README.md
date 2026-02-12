# AI Hedge Fund

AI-native hedge fund using a multi-agent LLM system with real market data. Built with LangGraph for agent orchestration and Polygon.io for market data.

**This project is for educational and research purposes only. Not financial advice.**

## How It Works

```
Market Data (Polygon.io)
        |
        v
┌─────────────────────────────────────────────┐
│         Analyst Agents (parallel)            │
│                                              │
│  Core Analysts         Persona Agents        │
│  ├── Fundamentals      ├── Warren Buffett    │
│  ├── Technical         └── (more coming)     │
│  ├── Sentiment                               │
│  ├── Valuation                               │
│  └── Growth                                  │
└──────────────────────┬──────────────────────-┘
                       |
                       v
┌─────────────────────────────────────────────┐
│              Risk Manager                    │
│   Consensus voting, volatility penalty,      │
│   exposure caps, position sizing             │
└──────────────────────┬──────────────────────-┘
                       |
                       v
┌─────────────────────────────────────────────┐
│            Portfolio Manager                  │
│   Signal-to-trade, confidence-based sizing   │
└──────────────────────────────────────────────┘
```

### Dual Mode

- **Rule-based (default)**: Deterministic scoring — works without any LLM API key
- **LLM-powered (`--use-llm`)**: Same data, but an LLM reasons about the facts for nuanced signals
- **Persona agents (`--personas`)**: Famous investor personas (Buffett, etc.) that always use LLM

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/DanisHack/ai-hedge-fund.git
cd ai-hedge-fund
pip install -e .
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env with your API keys
```

**Required:**
- `POLYGON_API_KEY` — for market data ([free tier available](https://polygon.io))

**Optional (for LLM features):**
- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`, or `DEEPSEEK_API_KEY`

### 3. Run analysis

```bash
# Rule-based (no LLM key needed)
python -m src.main --ticker AAPL,MSFT,NVDA

# With LLM reasoning
python -m src.main --ticker AAPL,MSFT,NVDA --use-llm

# With Warren Buffett persona
python -m src.main --ticker AAPL --personas buffett --use-llm
```

### 4. Backtest

```bash
python -m src.backtester \
  --ticker AAPL,MSFT,NVDA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31
```

## CLI Options

### Analysis (`src.main`)

| Flag | Default | Description |
|------|---------|-------------|
| `--ticker / -t` | required | Comma-separated tickers |
| `--start-date` | 90 days ago | Start date (YYYY-MM-DD) |
| `--end-date` | today | End date (YYYY-MM-DD) |
| `--cash` | 100000 | Starting cash |
| `--model` | gpt-4o-mini | LLM model name |
| `--provider` | openai | LLM provider |
| `--use-llm` | false | Enable LLM reasoning for analysts |
| `--personas` | none | Investor personas (e.g. `buffett` or `all`) |
| `--show-reasoning` | true | Log agent reasoning |
| `--debug` | false | Debug logging |

### Backtester (`src.backtester`)

All analysis flags above, plus:

| Flag | Default | Description |
|------|---------|-------------|
| `--frequency / -f` | weekly | `daily`, `weekly`, or `monthly` |
| `--lookback` | 90 | Lookback window in days per step |
| `--benchmark` | SPY | Benchmark ticker (`none` to disable) |
| `--commission` | 0.001 | Commission rate per trade (0.1%) |
| `--slippage` | 0.00005 | Slippage rate per trade (0.005%) |
| `--export` | none | Export to file (`results.json` or `results.csv`) |

#### Backtest Output

1. **Summary** — tickers, period, initial vs final value, P&L, transaction costs
2. **Performance Metrics** — total/annualized return, Sharpe ratio, max drawdown, volatility, Calmar ratio (with benchmark)
3. **Trade Statistics** — win rate, profit factor, avg win/loss
4. **Trade Log** — recent trades, color-coded
5. **Equity Curve** — ASCII chart

## Agents

### Core Analysts (always run)

| Agent | What It Analyzes |
|-------|-----------------|
| Fundamentals | Net margin, ROE, debt/equity, revenue growth |
| Technical | SMA crossover (20/50), RSI(14), volume trend, price vs SMA50 |
| Sentiment | News headline keyword scoring (24 positive + 24 negative keywords) |
| Valuation | DCF, P/E, P/B, free cash flow yield |
| Growth | Revenue/earnings growth rates, acceleration, consistency, margin expansion |

### Persona Agents (opt-in via `--personas`)

| Persona | Investment Style |
|---------|-----------------|
| Warren Buffett | Value investing — moats, margin of safety, ROE consistency, low debt |
| Ben Graham | Deep value — Graham Number, earnings stability, financial strength, net-net |

### Pipeline Agents

| Agent | Role |
|-------|------|
| Risk Manager | Consensus voting across analysts, volatility penalty, exposure caps |
| Portfolio Manager | Converts signals to buy/sell/hold with confidence-based position sizing |

## Project Structure

```
ai-hedge-fund/
├── src/
│   ├── agents/        # Analyst + persona + risk + portfolio agents
│   ├── backtest/      # Engine, portfolio tracker, metrics, export
│   ├── config/        # Settings, agent registry
│   ├── data/          # Polygon.io client, models, cache
│   ├── graph/         # LangGraph workflow & state
│   ├── llm.py         # LLM factory (5 providers) + structured output caller
│   ├── main.py        # CLI entry point
│   └── backtester.py  # Backtest CLI entry point
├── tests/             # 210 unit tests
├── pyproject.toml
└── README.md
```

## LLM Providers

| Provider | Model Example | Env Variable |
|----------|--------------|--------------|
| OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-sonnet-4-20250514 | `ANTHROPIC_API_KEY` |
| Groq | llama3-8b-8192 | `GROQ_API_KEY` |
| Google | gemini-pro | `GOOGLE_API_KEY` |
| DeepSeek | deepseek-chat | `DEEPSEEK_API_KEY` |

```bash
# Use a different provider
python -m src.main --ticker AAPL --use-llm --model claude-sonnet-4-20250514 --provider anthropic
```

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Past performance does not indicate future results
- Consult a licensed financial advisor for investment decisions

By using this software, you agree to use it solely for learning purposes.

## License

MIT License - see [LICENSE](LICENSE) for details.
