# AI-Prophets Forecasting Agent — Architecture

> Paste any Mermaid block into **[mermaid.live](https://mermaid.live)** to render.
> Last updated from codebase evaluation: May 2026.

---

## 1. System Context

How the agent fits in the Prophet Hacks forecasting competition and which external systems it talks to.

```mermaid
flowchart TB
    subgraph ORG["Organizer Platform (Prophet Arena)"]
        EVAL["Evaluation Server\nMay 17–31 window"]
        SCORE["Brier Scoring\n(rank = market_brier − our_brier) × completion"]
        EVAL --> SCORE
    end

    subgraph AGENT_DEPLOY["Our Deployment"]
        API["FastAPI Agent\nPOST /predict · GET /health\nagent/server.py"]
    end

    subgraph EXTERNAL["External Services"]
        KALSHI["Kalshi Public API\napi.elections.kalshi.com/trade-api/v2\n(no auth)"]
        TAVILY["Tavily Search API\nrecency + category queries"]
        OR["OpenRouter API\nLLM gateway"]
        DS["deepseek/deepseek-r1\n(MODEL_A)"]
        CL["anthropic/claude-sonnet-4-5\n(MODEL_B)"]
        OR --> DS
        OR --> CL
    end

    subgraph LOCAL["Local Dev & Benchmarking"]
        EVAL_PY["evaluate.py\n26-event sample dataset"]
        LIVE["predict_live.py → score_live.py\nKalshi markets closing soon"]
        EVAL_LIVE["eval_live.py\npredict + wait + score in one run"]
    end

    EVAL -->|"POST Event JSON"| API
    API -->|"p_yes or probabilities + rationale"| EVAL

    API --> KALSHI
    API --> TAVILY
    API --> OR

    EVAL_PY --> API
    LIVE --> API
    EVAL_LIVE --> API
    LIVE --> KALSHI
    EVAL_LIVE --> KALSHI

    style ORG fill:#1a1a2e,stroke:#7b68ee,color:#fff
    style AGENT_DEPLOY fill:#0f3460,stroke:#00d4aa,color:#fff
    style EXTERNAL fill:#16213e,stroke:#ffa500,color:#fff
    style LOCAL fill:#2d2d44,stroke:#58a6ff,color:#fff
```

---

## 2. Repository Layout & Module Map

```mermaid
flowchart LR
    subgraph ENTRY["HTTP Layer"]
        SRV["agent/server.py\nFastAPI · EventRequest\nPOST /predict · GET /health"]
    end

    subgraph CORE["Pipeline Core"]
        PIPE["agent/pipeline.py\npredict() orchestrator"]
        CLS["agent/event_classifier.py\nStage 0: binary | multi | numeric"]
        CAL["agent/calibration.py\nStage 5: shrink + clamp"]
    end

    subgraph EVIDENCE["Stage 1 — Evidence"]
        KAL["agent/evidence/kalshi.py\nget_binary_prior · get_multi_prior"]
        SRCH["agent/evidence/search.py\nTavily: news · stats · category · numeric"]
    end

    subgraph REASON["Stage 2–4 — Reasoning"]
        ENS["agent/reasoning/ensemble.py\nrun_ensemble · geo-mean · extremize"]
    end

    subgraph TOOLS["Evaluation Scripts (not in request path)"]
        EV["evaluate.py"]
        PL["predict_live.py"]
        SL["score_live.py"]
        EL["eval_live.py"]
        UFC["run_ufc_eval.py"]
    end

    SRV --> PIPE
    PIPE --> CLS
    PIPE --> KAL
    PIPE --> SRCH
    PIPE --> ENS
    PIPE --> CAL
    ENS --> CAL

    EV --> SRV
    PL --> SRV
    PL --> SL
    EL --> SRV
    UFC --> SRV

    style ENTRY fill:#4a9eff,stroke:#fff,color:#000
    style CORE fill:#0f3460,stroke:#00d4aa,color:#fff
    style EVIDENCE fill:#0f2027,stroke:#ffd700,color:#fff
    style REASON fill:#2d1f3d,stroke:#e94560,color:#fff
    style TOOLS fill:#2d2d44,stroke:#7b68ee,color:#fff
```

---

## 3. Request Lifecycle (Sequence)

End-to-end path for a single `POST /predict` call.

```mermaid
sequenceDiagram
    autonumber
    participant C as Client / Organizer
    participant S as server.py
    participant P as pipeline.py
    participant E as event_classifier
    participant K as evidence/kalshi
    participant T as evidence/search
    participant R as reasoning/ensemble
    participant Cal as calibration

    C->>S: POST /predict (EventRequest JSON)
    S->>P: predict(event.model_dump())

    P->>E: classify(event)
    E-->>P: ClassifiedEvent(type, outcomes)

    par Stage 1 — asyncio.gather
        alt binary
            P->>K: get_binary_prior(market_ticker)
            P->>T: search_recent_news(title)
            P->>T: search_stats_history(title)
            P->>T: search_category(title, category)
        else multi
            P->>K: get_multi_prior(market_ticker, outcomes)
            P->>T: search_recent_news · search_stats_history · search_category
        else numeric
            P->>K: get_multi_prior(market_ticker, outcomes)
            P->>T: search_recent_news · search_stats_history · search_numeric(title)
        end
    end

    P->>R: run_ensemble(event, priors, news, stats, category_info, event_type)

    par Stage 2–4 — dual models
        R->>R: _call_model(MODEL_A) via OpenRouter
        R->>R: _call_model(MODEL_B) via OpenRouter
    end
    R-->>P: p_yes or probabilities, evidence_quality, rationale

    alt binary
        P->>Cal: calibrate_binary(p_raw, prior, close_time, quality)
        Cal-->>P: p_final ∈ [0.01, 0.99]
        P-->>S: {p_yes, rationale}
    else multi / numeric
        P->>Cal: calibrate_multi(probs, priors, close_time, quality, outcomes)
        Cal-->>P: dict normalized to sum = 1
        P-->>S: {probabilities: [{market, probability}], rationale}
    end

    S-->>C: JSON response

    Note over S: On any exception → uniform fallback (0.5 or 1/N)
```

---

## 4. Prediction Pipeline (Detailed Flow)

Implementation-accurate view of all stages. **Numeric events** share the multi-outcome ensemble path; they differ only in Stage 1 search (`search_numeric` instead of `search_category`).

```mermaid
flowchart TD
    INPUT(["POST /predict\nagent/server.py\nEventRequest → pipeline.predict"])

    INPUT --> PARSE["Normalize event dict\ntitle · outcomes · rules\ncategory · close_time · tickers"]

    PARSE --> S0

    subgraph S0["Stage 0 — event_classifier.classify()"]
        C1{"len(outcomes) ≤ 2?"}
        C2{"≥60% outcomes match\nnumeric regex?"}
        BIN["BINARY\nEventType.BINARY"]
        NUM["NUMERIC\nEventType.NUMERIC"]
        MUL["MULTI\nEventType.MULTI"]

        C1 -->|yes| BIN
        C1 -->|no| C2
        C2 -->|yes| NUM
        C2 -->|no| MUL
    end

    BIN --> S1B
    MUL --> S1M
    NUM --> S1N

    subgraph S1B["Stage 1 — Binary evidence (parallel)"]
        KB["kalshi.get_binary_prior\nGET /markets/{market_ticker}\nmid = (yes_ask + yes_bid) / 2"]
        NB["search_recent_news\ndays=2"]
        SB["search_stats_history"]
        CB["search_category\nper-category query template"]
    end

    subgraph S1M["Stage 1 — Multi evidence (parallel)"]
        KM["kalshi.get_multi_prior\nGET /markets?event_ticker=…\nmatch subtitle → outcome, normalize"]
        NM["search_recent_news"]
        SM["search_stats_history"]
        CM["search_category"]
    end

    subgraph S1N["Stage 1 — Numeric evidence (parallel)"]
        KN["kalshi.get_multi_prior\n(same as multi)"]
        NN["search_recent_news"]
        SN["search_stats_history"]
        XN["search_numeric\nofficial count / tally queries"]
    end

    S1B --> AGG
    S1M --> AGG
    S1N --> AGG

    AGG["Structured context string\nkalshi_prior · news · stats · category/numeric"]

    AGG --> S234

    subgraph S234["Stages 2–4 — reasoning/ensemble.run_ensemble()"]
        PROMPT["_build_prompt()\n7-step superforecaster system prompt\n+ event + all evidence"]

        MA["MODEL_A (env)\ndefault: deepseek/deepseek-r1\nOpenRouter · temp 0.3 · 3 retries"]
        MB["MODEL_B (env)\ndefault: anthropic/claude-sonnet-4-5"]

        PAR["asyncio.gather both models"]
        PAR --> MA
        PAR --> MB

        BIN_COMB["Binary: geometric mean of odds\n+ _extremize if both >0.60 or <0.40"]
        MUL_COMB["Multi/Numeric: geo-mean per outcome\nclamp 0.01 · renormalize sum=1"]

        MA --> BIN_COMB
        MB --> BIN_COMB
        MA --> MUL_COMB
        MB --> MUL_COMB

        PROMPT --> PAR
    end

    BIN --> BIN_COMB
    MUL --> MUL_COMB
    NUM --> MUL_COMB

    BIN_COMB --> S5
    MUL_COMB --> S5

    subgraph S5["Stage 5 — calibration.py"]
        H["hours_to_close from close_time"]
        SF["_shrink_factor\n>2wk: 0.60 · 1–2wk: 0.40\n2–7d: 0.25 · <48h: 0.15\nWeak evidence: +0.20"]
        SH["p = (1−shrink)×p_raw + shrink×anchor\nanchor = Kalshi prior or 0.5 / 1/N"]
        CL["clamp 0.01–0.99 · multi renormalize"]

        H --> SF --> SH --> CL
    end

    CL --> OUT

    OUT(["Response\nBinary: {p_yes, rationale}\nMulti/Numeric: {probabilities[], rationale}\n\nNote: model_reasoning captured\ninternally but not returned to client"])

    style S0 fill:#0d1117,stroke:#58a6ff,color:#fff
    style S1B fill:#1a1a2e,stroke:#4a9eff,color:#fff
    style S1M fill:#1a1a2e,stroke:#4a9eff,color:#fff
    style S1N fill:#1a1a2e,stroke:#4a9eff,color:#fff
    style S234 fill:#0f3460,stroke:#00d4aa,color:#fff
    style S5 fill:#16213e,stroke:#ffa500,color:#fff
    style INPUT fill:#4a9eff,stroke:#fff,color:#000
    style OUT fill:#00d4aa,stroke:#fff,color:#000
```

---

## 5. Evaluation & Offline Tooling

Scripts used during development; none are invoked by the production `/predict` handler.

```mermaid
flowchart LR
    subgraph DATA["Data Sources"]
        DS1["ai-prophet-datasets\nsample-resolved tasks.jsonl\n(26 events)"]
        DS2["Kalshi open markets\nclosing in N minutes"]
    end

    subgraph RUN["Agent (must be running)"]
        AG["uvicorn agent.server:app\n:8000"]
    end

    subgraph SCRIPTS["Scripts"]
        E1["evaluate.py\nPOST all tasks · Brier vs baseline\n→ eval_results.json"]
        E2["predict_live.py\ndiscover closing markets · predict\n→ predictions_*.json\nschedule score_live"]
        E3["score_live.py\nfetch Kalshi result · Brier vs market"]
        E4["eval_live.py\npredict + poll until settled\n(one-shot live eval)"]
        E5["run_ufc_eval.py\nhardcoded UFC tickers · scheduled score"]
    end

    DS1 --> E1
    DS2 --> E2
    DS2 --> E4
    DS2 --> E5

    E1 --> AG
    E2 --> AG
    E4 --> AG
    E5 --> AG

    E2 --> E3
    E5 --> E3

    E3 --> DS2

    style DATA fill:#2d2d44,stroke:#7b68ee,color:#fff
    style RUN fill:#0f3460,stroke:#00d4aa,color:#fff
    style SCRIPTS fill:#16213e,stroke:#ffa500,color:#fff
```

---

## 6. External API Contracts

| Integration | Endpoint / SDK | Auth | Used by |
|---|---|---|---|
| Kalshi markets | `GET …/trade-api/v2/markets/{ticker}` | None | `kalshi.get_binary_prior`, live scoring |
| Kalshi event markets | `GET …/markets?event_ticker=&status=open` | None | `kalshi.get_multi_prior`, `predict_live` |
| Tavily search | `AsyncTavilyClient.search()` | `TAVILY_API_KEY` | `evidence/search.py` (4 query types) |
| OpenRouter chat | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` | `ensemble._call_model` |
| Prophet Arena | Organizer → our `POST /predict` | `PA_SERVER_API_KEY` (registration) | Production eval window |

---

## 7. Technical Stack (as implemented)

| Component | Choice | Notes |
|---|---|---|
| HTTP server | FastAPI + uvicorn | `agent/server.py`, port 8000 |
| Async I/O | `asyncio.gather` | Evidence + dual LLM calls |
| LLM gateway | OpenAI SDK → OpenRouter | `MODEL_A`, `MODEL_B` env overrides |
| Web search | **Tavily** (`tavily-python`) | Not Brave — doc was outdated |
| Market prior | Kalshi public API | No keys required |
| Schemas / CLI | `ai-prophet-core` | In `requirements.txt`; agent code uses local Pydantic models |
| Config | `python-dotenv` | `.env` from `.env.example` |

---

## 8. File Structure

```
AI-Prophets-Forecasting404/
├── agent/
│   ├── server.py              # FastAPI — POST /predict, GET /health
│   ├── pipeline.py            # Orchestrates stages 1 → 2–4 → 5
│   ├── event_classifier.py    # Stage 0 routing
│   ├── calibration.py         # Stage 5 shrinkage
│   ├── evidence/
│   │   ├── kalshi.py          # Binary + multi priors
│   │   └── search.py          # Tavily: news, stats, category, numeric
│   └── reasoning/
│       └── ensemble.py        # Dual-model prompt, JSON parse, ensemble
├── evaluate.py                # Offline Brier on 26-event dataset
├── predict_live.py            # Live Kalshi window → predictions JSON
├── score_live.py              # Score predictions after settlement
├── eval_live.py               # End-to-end live eval in one process
├── run_ufc_eval.py            # One-off UFC event batch
├── requirements.txt
├── .env.example
├── CONTEXT.md                 # Hackathon context & decisions
└── architecture.md            # This document
```

---

## 9. Codebase Evaluation Summary

### Strengths

- **Clear stage separation**: classifier → evidence → ensemble → calibration maps cleanly to modules.
- **Parallel I/O**: Stage 1 and dual-model calls minimize latency per request.
- **Calibration aligned with scoring**: Time-to-close shrinkage and weak-evidence penalty directly target Brier (avoid overconfidence).
- **Resilient HTTP layer**: `/predict` catches exceptions and returns neutral fallbacks so completion rate stays high.
- **Operational tooling**: `evaluate.py` plus live Kalshi scripts support iterative benchmarking.

### Gaps vs original design doc

| Design intent | Actual behavior |
|---|---|
| Brave Search API | **Tavily** is wired in `search.py` |
| Separate numeric LLM handler | Numeric uses **same multi ensemble**; only search query differs |
| `ai-prophet-core` schemas in agent | Agent defines its own `EventRequest`; core package unused in runtime path |
| Return full reasoning chain | Only `rationale` returned; `model_reasoning` dropped before response |
| Stage 0 “parse” step | Parsing is implicit in Pydantic + `classify()` |

### Evaluation SLA (organizer guidelines)

During the live evaluation window:

| Parameter | Value | Implication for this agent |
|---|---|---|
| HTTP timeout per request | **10 minutes** | Pipeline (~1–3 min typical) fits comfortably; no need to strip models for speed |
| Organizer retries | **None** | A failed or timed-out request is final from Arena’s side |
| In-agent retries | **Allowed** within the 10 min window | Existing 3× LLM retries in `ensemble.py` are appropriate |
| Event cadence | **Sequential, ~1 event / 10 min** | No concurrent load; single uvicorn worker is fine; no queue/backpressure design required |

**Architecture takeaway:** Latency is **not** a ranking bottleneck under these rules. Optimize for **Brier quality** (evidence, calibration, multi-label handling) rather than sub-minute responses. The 30s default in `ai-prophet-core`’s `ServerAPIClient` applies to CLI/dev tooling, not the Arena evaluation caller.

### Risk register (verified against [ai-prophet](https://github.com/ai-prophet/ai-prophet))

Ranking formula from hackathon context: **`(market_avg_brier − our_avg_brier) × completion_rate`**.

| ID | Risk | Severity | Status | Notes |
|---|---|---|---|---|
| R1 | **Request latency vs organizer timeout** | 🟢 Low | **Resolved** | Arena allows **10 min** per request; events arrive **~1 per 10 min**. Typical pipeline 60–120s is well within budget. |
| R2 | **Multi-label events forced to sum=1** | 🔴 Critical | **Mitigated** | `is_multi_label()` heuristic in `event_classifier.py`; pipeline/ensemble/calibration/fallback skip renormalize when multi-label. Pending organizer confirmation. |
| R3 | **Response contract ambiguity** | 🟠 High | Confirmed | Official [setup skill](https://github.com/ai-prophet/ai-prophet/tree/main/skills/ai-prophet-setup) documents HTTP response as **`p_yes` only**. Agent returns `{probabilities: [...]}` for 3+ outcomes. Datasets clearly send 3–35 `outcomes` per event — verify Arena accepts `probabilities` before eval window. |
| R4 | **Silent error fallback** | 🟠 High | Trade-off | `server.py` returns HTTP 200 with `p_yes=0.5` or uniform `1/N` on any exception. Arena **won’t retry** — a fallback is a **permanent** 0.25-class Brier hit for that event. Prefer exhausting in-agent retries (within 10 min) before returning fallback. |
| R5 | **`rationale` length limit** | 🟡 Medium | **Resolved** | `truncate_rationale()` in `agent/response_utils.py`; applied in `server.py` `_finalize_response` before every `/predict` response. |
| R6 | **Multi-outcome Kalshi prior matching** | 🟡 Medium | Confirmed | `get_multi_prior` needs **≥2** fuzzy subtitle matches. League-winner events (18–20 teams) often fail → prior `None` → heavy shrink to `1/N`, reducing edge vs market. |
| R7 | **`p_yes` = P(outcomes[0])** | 🟢 Low (if rules aligned) | Mitigated | Official datasets encode rules as “If {outcomes[0]} … resolves to Yes” ([dataset docs](https://github.com/ai-prophet/ai-prophet/blob/main/docs/using_sample_datasets.md)). Matches `evaluate.py` scoring. **Risk if** organizer sends `["Yes","No"]` with non-standard rule ordering. |
| R8 | **Large-outcome LLM coverage** | 🟡 Medium | Confirmed | 20–35 outcome events: prompt lists all outcomes; models may omit some in JSON. `_extract_multi_probs` only needs ≥2 parsed; missing outcomes get `0.01` default — silent distortion. |
| R9 | **No `close_time` guard** | 🟡 Medium | Confirmed | `prophet forecast predict` **skips past-deadline events** per official docs. Agent does not — will still burn API $ and return stale forecasts if called late. |
| R10 | **Registration / deployment** | 🟠 High (ops) | Open | Requires `prophet forecast register --endpoint-url <url>` + `PA_SERVER_API_KEY`. `server.py` uses `reload=True` in `main()` — not production-safe. |
| R11 | **LLM JSON / single-model fallback** | 🟡 Medium | Confirmed | 3 retries per model fits the 10 min window. If both models fail, falling back to 0.5/1/N is costly (no organizer retry). Consider more retries or a cheaper backup model before neutral fallback. |
| R12 | **Numeric vs categorical buckets** | 🟡 Medium | Confirmed | [sample-economics](https://github.com/ai-prophet/ai-prophet/blob/main/docs/using_sample_datasets.md) uses range buckets. Regex classifier may miss labels → treated as MULTI with sum=1 prior (winner-take-all assumption). |

**Removed / downgraded from initial review:**
- “Brave vs Tavily” — not a hackathon risk; implementation choice is fine if Tavily works.
- “Binary `p_yes` semantics” — **low risk** given official dataset rule pattern (outcomes[0] = YES side).
- “Latency / concurrency” — **low risk** given 10 min timeout and sequential ~1 event / 10 min cadence.

---

## 10. Scoring Reminder

```
Final rank score = (market_avg_brier − our_avg_brier) × completion_rate

Brier (binary)  = (p_yes − actual)²     where actual ∈ {0, 1}
Brier (multi)   = (1/N) Σ (p_i − actual_i)²

Never be overconfident — quadratic penalty dominates ranking
```

---

## 11. Competitive Edge (design intent)

```
Market price     = crowd wisdom (often stale by hours)
Our edge         = last 24–48h news (Tavily, days=2 on recency query)
                 + structured 7-step reasoning (both models)
                 + geometric mean of odds (ensemble)
                 + time/evidence calibration toward Kalshi prior
```
