# AI-Prophets-Forecasting404 — Full Project Context

> This document gives a collaborating AI (Cursor) full context on every decision made
> so far. Read this entirely before writing any code or suggesting changes.

---

## What We're Building

A **forecasting agent** for the Prophet Hacks hackathon (May 16–17, 2026, UChicago).
We are competing in the **Forecasting Track** (not Trading Track).

The agent is a **FastAPI server** with a single `POST /predict` endpoint.
During the 2-week evaluation window (May 17–31), the organizer's server
hits our endpoint with real-world events and scores our probability estimates
using **Brier score** (lower = better).

**Final ranking score = (market_avg_brier − our_avg_brier) × completion_rate**
We need to beat the Kalshi prediction market crowd. Higher score = better rank.

---

## How Evaluation Works

1. Organizer sends one event at a time to `POST /predict`
2. We return a probability estimate
3. When the real-world event resolves, they compute `(our_prob − actual)²`
4. Average across ~200 events over 2 weeks = our Brier score
5. **Ground truth = real-world outcome** (NOT Kalshi's prediction)
6. We register our deployed endpoint via: `prophet forecast register --endpoint-url <url>`

---

## Event Format (what our endpoint receives)

```json
{
  "event_ticker": "KXNBAGAME-26MAY15DETCLE",
  "market_ticker": "KXNBAGAME-26MAY15DETCLE",
  "title": "Will Cleveland beat Detroit in NBA Game 6 on May 15?",
  "description": "Resolves YES if Cleveland wins...",
  "category": "Sports",
  "rules": "If Cleveland wins... then the market resolves to Yes.",
  "close_time": "2026-05-15T20:00:00Z",
  "outcomes": ["Cleveland", "Detroit"],
  "resolved_outcome": null
}
```

**Three event types we discovered from the real dataset:**

| Type | Outcomes example | Response format |
|---|---|---|
| Binary | `["Yes","No"]` or `["Cleveland","Detroit"]` | `{p_yes: 0.67}` |
| Multi-outcome | `["PSG","Lille","Monaco"...]` (up to 30) | `{probabilities: [{market, probability}...]}` |
| Numeric range | `["0","1","2"..."9"]` or `["49 or below","50","51"...]` | `{probabilities: [{market, probability}...]}` |

**Key insight:** `market_ticker` is ALWAYS a Kalshi ticker (starts with `KX`).
For binary events it's the direct market ticker. For multi-outcome events it's the event ticker.

---

## Response Format

```python
# Binary event — MUST use this format
{"p_yes": 0.67, "rationale": "full reasoning chain..."}

# Multi-outcome — MUST use this format
{
  "probabilities": [
    {"market": "Barcelona", "probability": 0.55},
    {"market": "Real Madrid", "probability": 0.30},
    {"market": "Atletico Madrid", "probability": 0.15}
  ],
  "rationale": "full reasoning chain..."
}
# probabilities MUST sum to 1.0
# each probability MUST be between 0.01 and 0.99
```

---

## Our Architecture — 5 Stages

### Stage 0 — Event Type Router
```python
len(outcomes) == 2                          → BINARY
len(outcomes) > 2, labels look numeric      → NUMERIC
len(outcomes) > 2, labels are names/strings → MULTI
```
Numeric detection: regex match on outcomes like "0","1",...,"9" or "49 or below", "60 or above"

---

### Stage 1 — Parallel Evidence Gathering (asyncio.gather — all 4 at once)

**① Kalshi Price (FREE public API — NO auth needed — confirmed working)**
```
Base URL: https://api.elections.kalshi.com  (confirmed, tested)
Binary:   GET /trade-api/v2/markets/{market_ticker}
          → prior = (yes_ask_dollars + yes_bid_dollars) / 2
Multi:    GET /trade-api/v2/markets?event_ticker={market_ticker}&status=open
          → price per outcome → normalize to sum=1
Fallback: if not found → prior = None → calibration uses 0.5 or 1/N
```
Price fields in response: `yes_ask_dollars`, `yes_bid_dollars`, `last_price_dollars` (string format, e.g. "0.65")

**② Brave Search — Recency (last 24-48h)**
```
Query: f"{title} news today"
Endpoint: GET https://api.search.brave.com/res/v1/web/search
Headers: X-Subscription-Token: {BRAVE_API_KEY}, Accept: application/json
Params: q={query}, count=5, freshness=pd
→ returns snippets of recent news
```

**③ Brave Search — Stats/History**
```
Query: f"{title} odds statistics history"
Same endpoint, no freshness filter
→ base rates, historical patterns
```

**④ Category-Specific Brave Search**
```
Sports    → f"{title} injury report team form head to head"
Economics → f"{title} analyst forecast consensus"
Elections → f"{title} polling forecast"
Entmt     → f"{title} prediction award odds"
```

---

### Stage 2+3+4 — Dual Model Reasoning (two parallel LLM calls)

Both models (via OpenRouter) receive identical context and must follow this
structured reasoning protocol in one call:

```
1. DECOMPOSE: Break event into 3-5 sub-questions, answer each from evidence
2. ARGUE FOR: Top 3 arguments supporting primary/YES outcome
3. ARGUE AGAINST: Top 3 arguments against primary/YES outcome
   Weight each argument by: recency > source quality > specificity
4. ESTIMATE: Final probability as JSON
```

**Model A:** `deepseek/deepseek-r1` — chain-of-thought reasoning
**Model B:** `anthropic/claude-sonnet-4-5` — calibrated nuanced reasoning

Both called in parallel via `asyncio.gather`.

**Combining outputs — Geometric Mean of Odds:**
```python
odds_a = p_a / (1 - p_a)
odds_b = p_b / (1 - p_b)
combined_odds = math.sqrt(odds_a * odds_b)
p_ensemble = combined_odds / (1 + combined_odds)
```

**Extremize if models agree:**
```python
if p_a > 0.60 and p_b > 0.60:
    p_ensemble = p_ensemble + 0.05 * (p_ensemble - 0.5)  # push further toward YES
if p_a < 0.40 and p_b < 0.40:
    p_ensemble = p_ensemble + 0.05 * (p_ensemble - 0.5)  # push further toward NO
```

**For multi-outcome:** apply geometric mean per outcome, then renormalize.

---

### Stage 5 — Calibration

**Time-to-close shrinkage** (shrink toward market prior or neutral):
```python
hours_remaining = (close_time - now).total_seconds() / 3600
if hours_remaining > 336:   # > 2 weeks
    shrink = 0.60
elif hours_remaining > 168: # 1-2 weeks
    shrink = 0.40
elif hours_remaining > 48:  # 2-7 days
    shrink = 0.25
else:                       # < 48 hours
    shrink = 0.15

p_calibrated = (1 - shrink) * p_ensemble + shrink * prior
# prior = kalshi_mid if available, else 0.5 for binary, 1/N for multi
```

**Evidence quality adjustment** (model reports Strong/Moderate/Weak):
```python
if evidence_quality == "Weak":
    shrink += 0.20  # extra shrink when evidence is weak
```

**Brier-aware clamping:**
```python
# Binary
p_yes = max(0.01, min(0.99, p_calibrated))

# Multi-outcome
probs = {k: max(0.01, v) for k, v in probs.items()}
total = sum(probs.values())
probs = {k: v / total for k, v in probs.items()}  # renormalize
```

---

## Tech Stack

```
Language:    Python 3.11+
Server:      FastAPI + uvicorn
LLM:         OpenAI SDK → OpenRouter (base_url=https://openrouter.ai/api/v1)
Model A:     deepseek/deepseek-r1
Model B:     anthropic/claude-sonnet-4-5
Market data: Kalshi public REST API (no auth)
Web search:  Brave Search API
SDK:         ai-prophet-core (for Event/Submission schemas)
```

---

## Environment Variables

```bash
OPENROUTER_API_KEY=sk-or-...       # required — all LLM calls
BRAVE_API_KEY=BSA...               # required — web search
MODEL_A=deepseek/deepseek-r1       # optional override
MODEL_B=anthropic/claude-sonnet-4-5 # optional override
# PA_SERVER_API_KEY=prophet_...    # when organizers distribute
```

---

## File Structure to Build

```
Forecasting_Agent/
├── agent/
│   ├── __init__.py
│   ├── server.py           ← FastAPI app, POST /predict
│   ├── pipeline.py         ← orchestrates all 5 stages
│   ├── event_classifier.py ← Stage 0: binary/multi/numeric
│   ├── evidence/
│   │   ├── __init__.py
│   │   ├── kalshi.py       ← Stage 1A: Kalshi price (public API)
│   │   └── search.py       ← Stage 1B/C/D: Brave search (3 queries)
│   ├── reasoning/
│   │   ├── __init__.py
│   │   └── ensemble.py     ← Stage 2+3+4: dual model reasoning
│   └── calibration.py      ← Stage 5: calibration + clamping
├── .env                    ← keys (gitignored)
├── .env.example
├── .gitignore
├── requirements.txt
└── architecture.md
```

---

## Key Decisions Already Made (don't revisit)

1. **Kalshi = no auth needed** — public API confirmed working at `api.elections.kalshi.com`
2. **Two models, not one** — geometric mean is mathematically better than averaging
3. **Stages 2+3+4 combined into one LLM call** — reduces latency, keeps reasoning coherent
4. **Binary returns `p_yes`, multi returns `probabilities` list** — matches the SDK contract
5. **Calibration shrinks toward prior, not toward 0.5** — prior (Kalshi) is smarter than neutral
6. **`asyncio.gather` for Stage 1** — all 4 evidence sources run in parallel
7. **OpenRouter** — gives access to DeepSeek R1 + Claude Sonnet with one API key

---

## What Still Needs to Be Built (current status)

- [ ] `agent/event_classifier.py`
- [ ] `agent/evidence/kalshi.py`
- [ ] `agent/evidence/search.py`
- [ ] `agent/reasoning/ensemble.py`
- [ ] `agent/calibration.py`
- [ ] `agent/pipeline.py`
- [ ] `agent/server.py`
- [ ] Deployment (Railway / Render / Fly.io)
- [ ] Register endpoint: `prophet forecast register --endpoint-url <url>`

---

## Important Rules from the SDK

- `p_yes` must be between **0.01 and 0.99** (not 0 or 1) — will be rejected otherwise
- `close_time` is UTC — skip events whose close_time is already past
- `market_ticker` is the stable ID used for scoring — preserve it exactly
- `rules` field = literal resolution criterion — feed verbatim to LLM, don't paraphrase
- Multi-outcome `probabilities` must sum to 1.0 — always renormalize before returning

---

## Collaboration Notes

- This repo is **AI-Prophets-Forecasting404** on GitHub (NiharikaShekar)
- Claude Code (me) designed the architecture and will write/review code
- Cursor AI (teammate) is also contributing — check each other's code
- We are in a 32-hour hackathon — ship working code, iterate fast
- **Brier score over cleverness** — correct calibration beats fancy features
