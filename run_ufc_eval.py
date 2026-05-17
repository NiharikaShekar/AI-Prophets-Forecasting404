"""
One-shot script: predict on the 6 live UFC events, save predictions,
and schedule score_live.py to run in 86 minutes (11:45 PM CT).
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta

import httpx

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
AGENT_URL   = "http://localhost:8000"

# Exact event tickers confirmed live at 10:19 PM CT May 16 2026
LIVE_EVENTS = [
    "KXUFCFIGHT-26MAY16DIAPER",
    "KXUFCFIGHT-26MAY16ROUCAR",
    "KXUFCMOF-26MAY16DIAPER",
    "KXUFCMOF-26MAY16ROUCAR",
    "KXUFCROUNDS-26MAY16DIAPER",
    "KXUFCROUNDS-26MAY16ROUCAR",
]


def _mid(market: dict) -> float:
    ask  = float(market.get("yes_ask_dollars") or market.get("yes_ask") or 0)
    bid  = float(market.get("yes_bid_dollars") or market.get("yes_bid") or 0)
    last = float(market.get("last_price_dollars") or market.get("last_price") or 0)
    if ask > 1: ask /= 100
    if bid > 1: bid /= 100
    if last > 1: last /= 100
    if ask > 0 and bid > 0:
        return round((ask + bid) / 2, 4)
    return round(last or ask, 4)


async def fetch_event(client: httpx.AsyncClient, event_ticker: str) -> dict | None:
    r = await client.get(f"{KALSHI_BASE}/events/{event_ticker}")
    if r.status_code != 200:
        print(f"  ERROR fetching {event_ticker}: {r.status_code}")
        return None

    event_data  = r.json().get("event", {})
    markets_raw = r.json().get("markets", [])

    # Fetch price for each market leg
    outcomes = []
    tickers  = []
    kalshi_prices = {}

    for mkt in markets_raw:
        ticker = mkt.get("ticker", "")
        r2 = await client.get(f"{KALSHI_BASE}/markets/{ticker}")
        if r2.status_code != 200:
            continue
        md     = r2.json().get("market", r2.json())
        result = md.get("result", "")
        status = md.get("status", "")
        if result or status != "active":
            continue

        # Outcome label: use subtitle, or last segment of ticker
        label = (md.get("subtitle") or ticker.split("-")[-1]).strip()
        price = _mid(md)
        outcomes.append(label)
        tickers.append(ticker)
        if price > 0:
            kalshi_prices[label] = price

    if len(outcomes) < 2:
        print(f"  SKIP {event_ticker}: only {len(outcomes)} active outcome(s)")
        return None

    competition = event_data.get("product_metadata", {}).get("competition", "UFC")
    title       = event_data.get("title", event_ticker)
    subtitle    = event_data.get("sub_title", "")
    rules       = (event_data.get("rules_primary") or
                   f"Predict the outcome of {title}. {subtitle}").strip()
    close_time  = markets_raw[0].get("close_time", "") if markets_raw else ""

    is_binary = len(outcomes) == 2

    return {
        "event_ticker":   event_ticker,
        "market_ticker":  event_ticker,
        "title":          title,
        "subtitle":       subtitle,
        "category":       "Sports",
        "rules":          rules,
        "close_time":     close_time,
        "outcomes":       outcomes,
        "tickers":        tickers,
        "is_binary":      is_binary,
        "kalshi_price":   kalshi_prices.get(outcomes[0]) if is_binary else None,
        "kalshi_prices":  kalshi_prices if not is_binary else None,
        "competition":    competition,
    }


async def predict_event(client: httpx.AsyncClient, ev: dict) -> dict | None:
    payload = {k: ev[k] for k in
               ["event_ticker", "market_ticker", "title", "category",
                "rules", "close_time", "outcomes"]}
    try:
        r = await client.post(f"{AGENT_URL}/predict", json=payload, timeout=120.0)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"  AGENT ERROR: {e}")
        return None


def schedule_scoring(predictions_file: str, delay_seconds: int):
    script  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_live.py")
    log     = predictions_file.replace("predictions_", "score_log_").replace(".json", ".log")
    cmd     = (f'nohup bash -c "sleep {delay_seconds} && '
               f'{sys.executable} {script} --file {predictions_file}" '
               f'> {log} 2>&1 &')
    subprocess.Popen(cmd, shell=True)
    run_at = datetime.now() + timedelta(seconds=delay_seconds)
    print(f"\n  score_live.py scheduled → {run_at.strftime('%I:%M %p')} CT")
    print(f"  Log → {log}")


async def main():
    now_ct = datetime.now(timezone.utc) - timedelta(hours=5)
    print(f"CT time : {now_ct.strftime('%I:%M %p')} May 16, 2026")
    print(f"Events  : {len(LIVE_EVENTS)} UFC fights")
    print(f"Agent   : {AGENT_URL}\n")

    async with httpx.AsyncClient(timeout=15.0) as client:
        # ── Phase 1: Fetch Kalshi event details ───────────────────────────
        print("Fetching event details from Kalshi...")
        events = []
        for et in LIVE_EVENTS:
            ev = await fetch_event(client, et)
            if ev:
                events.append(ev)
                print(f"  ✓ {ev['title']}  ({len(ev['outcomes'])} outcomes)")
        print()

        if not events:
            print("No events fetched — check Kalshi API.")
            return

        # ── Phase 2: Run agent predictions ────────────────────────────────
        print(f"Running agent on {len(events)} events...\n{'─'*65}")
        predictions = []
        for i, ev in enumerate(events, 1):
            print(f"[{i}/{len(events)}] {ev['title']}")
            if ev["is_binary"]:
                kp = ev.get("kalshi_price") or 0
                print(f"  Kalshi : {ev['outcomes'][0]} {kp:.1%}  vs  {ev['outcomes'][1]} {1-kp:.1%}")
            else:
                kprices = ev.get("kalshi_prices") or {}
                for o, p in kprices.items():
                    print(f"  Kalshi : {o:<20} {p:.1%}")

            prediction = await predict_event(client, ev)

            if prediction:
                if "p_yes" in prediction:
                    print(f"  Agent  : p_yes = {prediction['p_yes']:.1%}")
                elif "probabilities" in prediction:
                    for item in sorted(prediction["probabilities"],
                                       key=lambda x: -x["probability"])[:3]:
                        print(f"  Agent  : {item['market']:<20} {item['probability']:.1%}")

                # Print each model's reasoning
                model_reasoning = prediction.get("model_reasoning", {})
                for model_name, reasoning_text in model_reasoning.items():
                    if reasoning_text:
                        short_name = model_name.split("/")[-1]
                        print(f"\n  ── {short_name} reasoning ──")
                        # Print up to 800 chars so it's readable but not overwhelming
                        preview = reasoning_text.strip()[:800]
                        for line in preview.split("\n"):
                            print(f"    {line}")
                        if len(reasoning_text) > 800:
                            print(f"    ... [{len(reasoning_text)} chars total]")
            else:
                print("  Agent  : ERROR — no prediction")

            predictions.append({**ev, "our_prediction": prediction,
                                 "predicted_at": datetime.now(timezone.utc).isoformat()})
            print()

    # ── Phase 3: Save + schedule scoring ──────────────────────────────────
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"predictions_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(predictions, f, indent=2, default=str)

    print(f"{'─'*65}")
    print(f"Predictions saved → {out_file}")

    # Schedule 86 min from now (11:45 PM CT) — after all fights finish
    delay = 86 * 60
    print(f"\nScheduling score_live.py in {delay//60} min (11:45 PM CT)...")
    schedule_scoring(out_file, delay)
    print("\nDone. Come back at 11:45 PM CT for your Brier scores.")


if __name__ == "__main__":
    asyncio.run(main())
