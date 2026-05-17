# AI Forecasting Agent

A deliberative multi-agent system that predicts Kalshi prediction market outcomes by combining domain-specific evidence gathering, independent dual-model reasoning, and a judge LLM that arbitrates the final probability.

---

## The Problem

Kalshi prediction markets are efficient — the crowd price is the baseline. Every participant sees the same market price. Beating it requires something the crowd doesn't have: better evidence, more rigorous reasoning, and calibrated output that avoids the biases humans and single-model systems share.

The metric is **Brier Score** — `(predicted_probability − actual_outcome)²`. Lower is better. A perfect predictor scores 0. A naive 50/50 guess scores ~0.25.

---

## Architecture

### High-Level Pipeline

```
┌──────────────────────────────────────────┐
│           KALSHI EVENT INPUT             │
│  title · category · outcomes · close_time│
└─────────────────┬────────────────────────┘
                  │
        ┌─────────▼──────────┐
        │   Parallel gather  │
        └──┬──────┬───────┬──┘
           │      │       │
    ┌──────▼─┐ ┌──▼───┐ ┌─▼──────────┐
    │ Kalshi │ │Tavily│ │ Domain APIs│
    │  Prior │ │Search│ │ (8 sources)│
    └──────┬─┘ └──┬───┘ └─┬──────────┘
           └──────┴────────┘
                  │  Evidence String
        ┌─────────▼──────────────┐
        │    Run in parallel     │
        └──────┬────────────┬────┘
               │            │
    ┌──────────▼──┐  ┌──────▼────────┐
    │  Model A    │  │   Model B     │
    │ DeepSeek R1 │  │ Claude Sonnet │
    │ chain-of-   │  │ calibrated    │
    │ thought     │  │ reasoning     │
    └──────┬──────┘  └──────┬────────┘
           └────────┬────────┘
                    │ Both predictions + reasoning
          ┌─────────▼──────────┐
          │    JUDGE AGENT     │
          │   Claude Sonnet    │
          │ audits both models │
          │ catches correlated │
          │ errors & bias      │
          └─────────┬──────────┘
                    │ Final probability + critique
          ┌─────────▼──────────┐
          │    CALIBRATION     │
          │ Brier-optimal      │
          │ extremizing +      │
          │ time-to-close      │
          │ shrinkage          │
          └─────────┬──────────┘
                    │
          ┌─────────▼──────────┐
          │  FINAL PREDICTION  │
          │ p_yes · rationale  │
          │ critique · confidence│
          └────────────────────┘
```

### Evidence Layer — Category-Aware Routing

```
┌──────────────────────────────┐
│        KALSHI EVENT          │
│   category · title · rules   │
└──────────────┬───────────────┘
               │
       ┌───────▼────────┐
       │   DISPATCHER   │
       │ keyword-based  │
       │ auto-classify  │
       └──┬───────────┬─┘
          │           │ always
          │     ┌─────▼──────┐
          │     │   TAVILY   │
          │     │ web search │
          │     └────────────┘
          │ route by category
    ┌─────▼──────────────────────────────────────────────┐
    │                                                    │
  Sports   Crypto   Elections  Economics  Financials     │
  The      Coin-    Polymarket FRED       FRED +         │
  Odds API Gecko +  Gamma API  (macro)    SEC EDGAR      │
           Fear &                                        │
           Greed                                         │
    │                                                    │
  Companies  Climate  Culture      Mentions              │
  SEC EDGAR  NOAA CDO TMDB +       Reddit OAuth          │
  + FRED     7-day    Reddit                             │
             weather                                     │
    └────────────────────────────────────────────────────┘
               │
     ┌─────────▼──────────┐
     │   EVIDENCE STRING  │
     │  all sources merged│
     │  → Model A + B     │
     └────────────────────┘
```

If the category field is missing or unrecognized, a keyword scorer runs over the event title and resolution rules, matches against 12 category buckets, and selects the best fit automatically. No crashes, no blind spots.

---

## What We Built

### 1. Three-Model Judge Pipeline

Most forecasting systems call one model and return its output. We built a deliberative pipeline:

- **Model A (DeepSeek R1)** — runs extended chain-of-thought reasoning using a structured 7-step framework: decompose → base rate → argue for → argue against → synthesize → calibrate → estimate
- **Model B (Claude Sonnet)** — independently runs the same framework, optimized for calibrated probabilistic output
- **Judge Agent (Claude Sonnet)** — sees both predictions *and* both full reasoning traces, then audits for anchoring bias, overconfidence, ignored evidence, and correlated errors

The judge's job is not to average A and B. Its job is to find the most defensible number given the evidence, which sometimes means overriding both models entirely. It produces a `confidence` rating and a ≤1000-character `critique` explaining its reasoning.

The geometric mean of odds is used as a fallback if the judge fails. Extremizing (pushing the combined probability further from 0.5 when both models agree) is applied on top.

### 2. Nine-Source Evidence Fusion

Evidence is gathered in parallel before any model sees the event. Sources are selected by category:

| Category | APIs |
|---|---|
| Sports | The Odds API — H2H implied probabilities |
| Crypto | CoinGecko market data + Alternative.me Fear & Greed Index |
| Elections | Polymarket Gamma API — crowd probabilities + volume |
| Economics | FRED — CPI, unemployment rate, Fed funds rate, GDP |
| Financials | FRED + SEC EDGAR full-text search |
| Companies | SEC EDGAR 8-K/10-K/10-Q filings |
| Climate | NOAA CDO — 7-day temperature and precipitation |
| Culture | TMDB movie/TV/person data + Reddit social buzz |
| Mentions | Reddit OAuth — post volume and sentiment |

Tavily web search runs on **every** event regardless of category, pulling the most recent news and historical context. All APIs degrade gracefully — a missing key returns an empty string and the pipeline continues.

### 3. Multi-Label Market Support

Kalshi has two distinct market structures that require different probability handling:

- **Winner-take-all**: exactly one outcome resolves YES. Probabilities must sum to 1.0.
- **Multi-label**: multiple outcomes can resolve YES simultaneously (e.g. "top 5 finishers", "films nominated for Best Picture").

The system detects multi-label markets using pattern matching on resolution rules and skips renormalization for them — each outcome probability is an independent marginal, not a share of a constrained distribution. This distinction is propagated through evidence gathering, both models' prompts, the judge, and calibration.

### 4. Brier-Optimal Calibration

Raw model output is not the final answer. A calibration stage adjusts for two known biases:

- **Time-to-close shrinkage**: as the event approaches resolution, uncertainty decreases and the market price becomes a stronger anchor. Predictions are shrunk toward the Kalshi prior proportionally.
- **Extremizing**: when both models agree on a direction, the combined probability is pushed further from 0.5. The intuition is that independent agreement is evidence the signal is real and not noise, so the ensemble should be more decisive than any single model.

---

## Project Structure

```
agent/
├── pipeline.py              # main predict() entry point
├── server.py                # FastAPI HTTP server
├── event_classifier.py      # binary / multi / numeric + multi-label detection
├── calibration.py           # time-to-close shrinkage + extremizing
├── fallback.py              # smart fallback using Kalshi market prices
├── response_utils.py        # rationale truncation + logging helpers
├── evidence/
│   ├── kalshi.py            # Kalshi market prior (bid/ask mid)
│   ├── search.py            # Tavily web search
│   ├── dispatcher.py        # category routing + keyword fallback
│   └── apis/
│       ├── odds.py          # The Odds API (sports)
│       ├── coingecko.py     # CoinGecko + Fear & Greed (crypto)
│       ├── fred.py          # FRED (economics, commodities, financials)
│       ├── polymarket.py    # Polymarket Gamma (elections, politics)
│       ├── reddit.py        # Reddit OAuth (mentions, culture)
│       ├── noaa.py          # NOAA CDO (climate)
│       ├── tmdb.py          # TMDB (culture)
│       └── edgar.py         # SEC EDGAR (companies, financials)
└── reasoning/
    ├── ensemble.py          # dual-model parallel calls + geo mean
    └── judge.py             # judge agent — audits A+B, sets final P
tests/
├── test_event_classifier.py
└── test_response_utils.py
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# fill in your API keys
```

**Required:**
- `OPENROUTER_API_KEY` — all LLM calls go through OpenRouter
- `TAVILY_API_KEY` — web search, called on every event

**Optional (each degrades gracefully if absent):**
- `ODDS_API_KEY` — sports markets
- `COINGECKO_API_KEY` — crypto (Fear & Greed works without a key)
- `FRED_API_KEY` — economics and commodities
- `REDDIT_CLIENT_ID` + `REDDIT_CLIENT_SECRET` — Reddit mentions
- `NOAA_TOKEN` — climate data
- `TMDB_API_KEY` — culture/entertainment

No key needed: Polymarket Gamma API, SEC EDGAR

```bash
# Run the server
uvicorn agent.server:app --reload

# Run tests
pytest tests/
```

---

## Models

Configured via environment variables:

```
MODEL_A=deepseek/deepseek-r1
MODEL_B=anthropic/claude-sonnet-4-5
MODEL_JUDGE=anthropic/claude-sonnet-4-5
```

All routed through [OpenRouter](https://openrouter.ai).

---

## Tech Stack

Python · FastAPI · asyncio · OpenRouter · Tavily · The Odds API · CoinGecko · FRED · Polymarket · SEC EDGAR · NOAA · TMDB · Reddit API · python-pptx
