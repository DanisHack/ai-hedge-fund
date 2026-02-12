# AI Hedge Fund

An AI-powered hedge fund that uses a team of autonomous agents to analyze stocks, manage risk, and make trading decisions. Each agent specializes in a different aspect of investing — from fundamental analysis to technical indicators to channeling the strategies of legendary investors like Warren Buffett and Michael Burry.

**This is for educational and research purposes only. Not financial advice.**

---

## What It Does

You give it a list of stock tickers. It runs 6 specialized analysts in parallel, filters their signals through a risk manager, and outputs concrete buy/sell/hold decisions with position sizes.

```bash
python -m src.main --ticker AAPL,MSFT,NVDA
```

That's it. No API keys needed for the default rule-based mode.

Want smarter analysis? Add an LLM:

```bash
python -m src.main --ticker AAPL,MSFT,NVDA --use-llm
```

Want Warren Buffett's take? Or Cathie Wood's? Pick your personas:

```bash
python -m src.main --ticker AAPL --personas buffett,wood --use-llm
```

---

## How It Works

The system runs as a pipeline. Every agent works independently in parallel, then their signals flow through risk management and into a final portfolio decision.

```
         ┌── Fundamentals ──┐
         ├── Technical ──────┤
         ├── Sentiment ──────┤
         ├── Valuation ──────┼──► Risk Manager ──► Portfolio Manager ──► Trades
         ├── Growth ─────────┤
         ├── Macro Regime ───┤
         └── Personas (opt) ─┘
```

**Step 1: Analysis.** Six core analysts each examine the stock from a different angle — financials, price action, news, valuation, growth trajectory, and macro environment. They all run at the same time.

**Step 2: Risk Management.** The risk manager collects every analyst's signal and applies constraints:
- Takes a vote across all analysts (bullish vs bearish vs neutral)
- Penalizes high-volatility stocks (>3% daily swings)
- Groups correlated stocks together and caps each group at 40% of portfolio
- Limits any single position to 25% and total exposure to 90%

**Step 3: Execution.** The portfolio manager converts the risk-adjusted signals into actual trades — buy, sell, or hold — with dollar amounts sized by confidence level.

Built with [LangGraph](https://github.com/langchain-ai/langgraph) for agent orchestration and [Polygon.io](https://polygon.io) for market data.

---

## Setup

### 1. Install

```bash
git clone https://github.com/DanisHack/ai-hedge-fund.git
cd ai-hedge-fund
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
```

Add your API keys to `.env`:

| Key | Required? | What it's for |
|-----|-----------|---------------|
| `POLYGON_API_KEY` | **Yes** | Market data — prices, financials, news ([free tier](https://polygon.io)) |
| `OPENAI_API_KEY` | No | LLM reasoning with `--use-llm` or `--personas` |
| `ANTHROPIC_API_KEY` | No | Alternative LLM provider |
| `GROQ_API_KEY` | No | Alternative LLM provider |
| `GOOGLE_API_KEY` | No | Alternative LLM provider |
| `DEEPSEEK_API_KEY` | No | Alternative LLM provider |

> The system works without any LLM key. The default rule-based mode uses deterministic scoring — no LLM calls, fast execution, reproducible results. Add an LLM key only if you want `--use-llm` or `--personas`.

### 3. Run

```bash
python -m src.main --ticker AAPL,MSFT,NVDA
```

---

## Three Ways to Use It

### 1. Single Analysis — "What should I do right now?"

Analyze stocks and get buy/sell/hold decisions:

```bash
# Rule-based analysis (fast, no LLM needed)
python -m src.main --ticker AAPL,MSFT,NVDA

# LLM-powered analysis (smarter, needs API key)
python -m src.main --ticker AAPL --use-llm

# With a specific investor's perspective
python -m src.main --ticker AAPL --personas buffett --use-llm

# Mix and match personas
python -m src.main --ticker AAPL --personas buffett,munger,burry --use-llm

# Use all 12 personas at once
python -m src.main --ticker AAPL --personas all --use-llm

# Pick your LLM provider
python -m src.main --ticker AAPL --use-llm --provider anthropic --model claude-sonnet-4-20250514
```

### 2. Backtesting — "How would this have performed?"

Test the strategy on historical data:

```bash
# Basic backtest over 2024
python -m src.backtester \
  --ticker AAPL,MSFT,NVDA \
  --start-date 2024-01-01 \
  --end-date 2024-12-31

# With downside protection
python -m src.backtester \
  --ticker AAPL,MSFT \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --stop-loss 0.10 \
  --trailing-stop 0.08 \
  --take-profit 0.20

# Export results for further analysis
python -m src.backtester \
  --ticker AAPL \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --frequency daily \
  --export results.json
```

**What you get:** summary panel, performance metrics (Sharpe ratio, max drawdown, Calmar ratio), trade statistics (win rate, profit factor), full trade log, and an ASCII equity curve — all compared against an SPY benchmark.

### 3. Paper Trading — "Let me track this live."

Run the strategy on real market data with a simulated portfolio:

```bash
# Run a trading cycle (creates or continues a portfolio)
python -m src.paper_trader run --ticker AAPL,MSFT,NVDA

# Check how your portfolio is doing
python -m src.paper_trader status

# Start fresh
python -m src.paper_trader reset --ticker AAPL,MSFT --cash 200000
```

Your portfolio is saved to `paper_portfolio.json` automatically. Run `paper_trader run` as often as you like — daily, weekly, whenever — and it picks up where you left off.

---

## The Agents

### Core Analysts (always run)

These six agents analyze every stock. They work by default without any LLM — add `--use-llm` for deeper reasoning.

| Agent | What it looks at | Key metrics |
|-------|-----------------|-------------|
| **Fundamentals** | Financial health and profitability | Net margin, ROE, debt/equity, revenue growth |
| **Technical** | Price action and momentum | SMA(20/50), RSI(14), MACD, Bollinger Bands, ADX, volume |
| **Sentiment** | Market mood from news | Headline scoring across 48 keyword categories |
| **Valuation** | Is it cheap or expensive? | DCF model, P/E, P/B, FCF yield, margin of safety |
| **Growth** | Is it growing faster or slower? | Revenue/earnings acceleration, consistency, margin expansion |
| **Macro Regime** | Where are we in the cycle? | Sector classification, cyclical vs defensive, sector momentum |

### Investor Personas (opt-in)

These are LLM-powered agents that analyze stocks through the lens of famous investors. Each one has a unique system prompt encoding that investor's philosophy, and a tailored set of financial facts fed to the LLM.

Use `--personas buffett` for one, `--personas buffett,munger,burry` for a few, or `--personas all` for everyone.

| Persona | Philosophy |
|---------|-----------|
| **Warren Buffett** | The Oracle of Omaha. Wonderful companies at fair prices. Looks for durable moats, consistent ROE, and margin of safety. |
| **Charlie Munger** | Buffett's partner. Only buys wonderful businesses at fair prices. Focuses on high ROIC, pricing power, and simple business models. |
| **Benjamin Graham** | The godfather of value investing. Buys hidden gems below intrinsic value. Graham Number, P/E < 15, P/B < 1.5, earnings stability. |
| **Aswath Damodaran** | The Dean of Valuation. Story must match numbers. DCF-driven, ROIC vs WACC, honest about uncertainty. |
| **Michael Burry** | The Big Short. Contrarian who hunts deep value. Tangible book value, FCF yield, asset-backed undervaluation. |
| **Peter Lynch** | Practical stock-picker. Seeks ten-baggers in everyday businesses. PEG ratio is king — P/E divided by growth should be below 1.0. |
| **Phil Fisher** | Meticulous growth investor. Deep scuttlebutt research. Revenue growth, margin expansion, heavy R&D reinvestment. |
| **Cathie Wood** | Queen of growth investing. Bets on disruptive innovation. Revenue growth acceleration, gross margins, massive TAM. |
| **Bill Ackman** | Activist investor. Takes bold positions and pushes for change. Margin improvement, FCF yield, operational turnarounds. |
| **Mohnish Pabrai** | The Dhandho investor. Heads I win big, tails I don't lose much. Asymmetric risk/reward at low cost. |
| **Rakesh Jhunjhunwala** | India's Big Bull. Earnings acceleration, market leadership, and bold conviction bets on growth. |
| **Stanley Druckenmiller** | Macro legend. Hunts for asymmetric opportunities with growth potential and macro tailwinds. |

---

## Risk Controls

### Automatic Stop-Loss and Take-Profit

Three types of downside protection, all checked automatically before each trading cycle:

| Type | What it does | Flag | Example |
|------|-------------|------|---------|
| **Fixed stop-loss** | Sells if price drops X% below your purchase price | `--stop-loss 0.10` | Exits at 10% loss |
| **Trailing stop** | Sells if price drops X% below its peak since purchase | `--trailing-stop 0.08` | Exits at 8% drop from highest point |
| **Take-profit** | Sells if price rises X% above your purchase price | `--take-profit 0.20` | Locks in at 20% gain |

If multiple triggers hit at once, priority is: fixed stop > trailing stop > take-profit. One sell per position per cycle.

### Correlation-Aware Sizing

The risk manager detects when stocks move together (correlation > 0.7 over 60 days) and groups them. Each correlated group is capped at 40% of the portfolio — so even if all 6 analysts love AAPL, MSFT, and GOOGL, the system won't put 75% into tech.

---

## CLI Reference

### All Flags

| Flag | Default | Used in | Description |
|------|---------|---------|-------------|
| `--ticker, -t` | *required* | all | Comma-separated tickers (e.g. `AAPL,MSFT,NVDA`) |
| `--cash` | `100000` | all | Starting capital in dollars |
| `--use-llm` | `false` | all | Use LLM reasoning for analyst agents |
| `--personas` | none | all | Investor personas (e.g. `buffett,munger` or `all`) |
| `--model` | `gpt-4o-mini` | all | LLM model name |
| `--provider` | `openai` | all | LLM provider (`openai`, `anthropic`, `groq`, `google`, `deepseek`) |
| `--show-reasoning` | `false` | all | Print each agent's reasoning to the console |
| `--debug` | `false` | all | Verbose debug logging |
| `--start-date` | *required* | backtest | Start date (`YYYY-MM-DD`) |
| `--end-date` | *required* | backtest | End date (`YYYY-MM-DD`) |
| `--frequency, -f` | `weekly` | backtest | Trading frequency: `daily`, `weekly`, or `monthly` |
| `--lookback` | `90` | backtest, paper | Days of history each agent looks at per step |
| `--benchmark` | `SPY` | backtest | Benchmark ticker for comparison (`none` to disable) |
| `--commission` | `0.001` | backtest, paper | Commission per trade (0.1%) |
| `--slippage` | `0.00005` | backtest, paper | Slippage per trade (0.005%) |
| `--stop-loss` | none | backtest, paper | Fixed stop-loss as decimal (e.g. `0.10` = 10%) |
| `--trailing-stop` | none | backtest, paper | Trailing stop as decimal (e.g. `0.08` = 8%) |
| `--take-profit` | none | backtest, paper | Take-profit as decimal (e.g. `0.20` = 20%) |
| `--export` | none | backtest | Export results to `.json` or `.csv` |
| `--state-file` | `paper_portfolio.json` | paper | Path to portfolio state file |

---

## Project Structure

```
ai-hedge-fund/
├── src/
│   ├── agents/              # All analyst, persona, risk, and portfolio agents
│   │   ├── fundamentals.py  #   Financial health scoring
│   │   ├── technical.py     #   Price action & indicators
│   │   ├── sentiment.py     #   News sentiment scoring
│   │   ├── valuation.py     #   DCF & relative valuation
│   │   ├── growth.py        #   Growth trajectory analysis
│   │   ├── macro_regime.py  #   Sector & cycle classification
│   │   ├── buffett.py       #   Warren Buffett persona
│   │   ├── graham.py        #   Ben Graham persona
│   │   ├── munger.py        #   Charlie Munger persona
│   │   ├── damodaran.py     #   Aswath Damodaran persona
│   │   ├── burry.py         #   Michael Burry persona
│   │   ├── lynch.py         #   Peter Lynch persona
│   │   ├── fisher.py        #   Phil Fisher persona
│   │   ├── wood.py          #   Cathie Wood persona
│   │   ├── ackman.py        #   Bill Ackman persona
│   │   ├── pabrai.py        #   Mohnish Pabrai persona
│   │   ├── jhunjhunwala.py  #   Rakesh Jhunjhunwala persona
│   │   ├── druckenmiller.py #   Stanley Druckenmiller persona
│   │   ├── risk_manager.py  #   Risk adjustment & position limits
│   │   └── portfolio_manager.py # Signal-to-trade conversion
│   ├── backtest/            # Backtesting engine
│   │   ├── engine.py        #   Date stepping & workflow execution
│   │   ├── portfolio_tracker.py # Position tracking, stop-loss
│   │   ├── metrics.py       #   Sharpe, drawdown, win rate, etc.
│   │   ├── models.py        #   Trade, Snapshot, StopLossConfig
│   │   └── export.py        #   JSON & CSV export
│   ├── paper_trading/       # Live paper trading
│   │   ├── runner.py        #   Single-cycle executor
│   │   └── state.py         #   Persistent portfolio state
│   ├── config/              # Configuration
│   │   ├── settings.py      #   API keys & defaults
│   │   └── agents.py        #   Agent registry
│   ├── data/                # Market data layer
│   │   ├── polygon_client.py #  Polygon.io API client
│   │   ├── models.py        #   Data models (Price, Metrics, etc.)
│   │   └── cache.py         #   TTL-based API response cache
│   ├── graph/               # LangGraph orchestration
│   │   ├── workflow.py      #   Workflow builder & runner
│   │   └── state.py         #   AgentState with merge logic
│   ├── llm.py               # LLM factory (5 providers)
│   ├── main.py              # Analysis CLI
│   ├── backtester.py        # Backtest CLI
│   └── paper_trader.py      # Paper trading CLI
├── tests/                   # 342 unit tests
├── .github/workflows/       # CI pipeline (Python 3.11/3.12/3.13)
├── pyproject.toml
└── .env.example
```

---

## Development

```bash
pip install -e ".[dev]"

# Run tests
python -m pytest tests/ -v

# Lint
python -m ruff check src/ tests/
```

**342 tests** covering all agents, the backtest engine, portfolio tracker, metrics, paper trading, export, and LLM integration. CI runs on every push across Python 3.11, 3.12, and 3.13.

---

## Disclaimer

This project is for **educational and research purposes only**.

- Not financial advice. Not intended for real trading.
- No guarantees of performance. Past results don't predict future returns.
- Consult a licensed financial advisor before making investment decisions.

## License

MIT — see [LICENSE](LICENSE).
