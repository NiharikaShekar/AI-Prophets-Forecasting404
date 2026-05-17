"""
Script 1 of 2 — Predict on Kalshi markets closing in 60-120 min.

Output: predictions_TIMESTAMP.json  (auto-scored by score_live.py in ~2 hrs)

Usage:
    python predict_live.py [--url http://localhost:8000] [--window-min 60] [--window-max 120]
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta

import httpx

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


# ── Price helpers ─────────────────────────────────────────────────────────────

def _parse_price(val) -> float:
    try:
        v = float(val)
        if v > 1:
            v = v / 100
        return max(0.01, min(0.99, v))
    except (TypeError, ValueError):
        return 0.0


def _mid(market: dict) -> float:
    yes_ask = _parse_price(market.get("yes_ask_dollars") or market.get("yes_ask"))
    yes_bid = _parse_price(market.get("yes_bid_dollars") or market.get("yes_bid"))
    if yes_ask > 0 and yes_bid > 0:
        return (yes_ask + yes_bid) / 2
    last = _parse_price(market.get("last_price_dollars") or market.get("last_price"))
    if last > 0:
        return last
    if yes_ask > 0:
        return yes_ask
    return 0.0


def _parse_close(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


# ── Event discovery ───────────────────────────────────────────────────────────

async def fetch_closing_soon(window_min: int, window_max: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    lo = now + timedelta(minutes=window_min)
    hi = now + timedelta(minutes=window_max)

    events: dict[str, list] = {}
    cursor = None

    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            params: dict = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            r = await client.get(f"{KALSHI_BASE}/markets", params=params)
            if r.status_code != 200:
                break
            body = r.json()
            markets = body.get("markets", [])
            if not markets:
                break
            for m in markets:
                ticker = m.get("ticker", "")
                if "MULTIVARIATE" in ticker.upper() or "MVE" in ticker.upper():
                    continue
                close_dt = _parse_close(m.get("close_time", ""))
                if close_dt and lo <= close_dt <= hi:
                    et = m.get("event_ticker") or ticker
                    events.setdefault(et, []).append(m)
            cursor = body.get("cursor")
            if not cursor:
                break

    records = []
    for event_ticker, ms in events.items():
        ms.sort(key=lambda m: m.get("ticker", ""))
        first = ms[0]
        close_dt = _parse_close(first.get("close_time", ""))

        if len(ms) == 1:
            kalshi_price = _mid(first)
            records.append({
                "event_ticker": event_ticker,
                "market_ticker": first.get("ticker", event_ticker),
                "title": first.get("title", event_ticker),
                "category": first.get("category") or "General",
                "rules": first.get("rules_primary", ""),
                "close_time": first.get("close_time", ""),
                "close_dt": close_dt.isoformat(),
                "outcomes": ["Yes", "No"],
                "is_binary": True,
                "n_markets": 1,
                # Kalshi crowd signal — for comparison in score_live.py
                "kalshi_price": kalshi_price,
                "kalshi_prices": None,
                "tickers": [first.get("ticker", "")],
            })
        else:
            subtitles = [m.get("subtitle") or m.get("title", f"Option {i+1}") for i, m in enumerate(ms)]
            raw = {sub: _mid(m) for sub, m in zip(subtitles, ms)}
            total = sum(raw.values())
            norm = {k: round(v / total, 4) for k, v in raw.items()} if total > 0 else {}
            records.append({
                "event_ticker": event_ticker,
                "market_ticker": event_ticker,
                "title": first.get("title", event_ticker),
                "category": first.get("category") or "General",
                "rules": first.get("rules_primary", ""),
                "close_time": first.get("close_time", ""),
                "close_dt": close_dt.isoformat(),
                "outcomes": subtitles,
                "is_binary": False,
                "n_markets": len(ms),
                "kalshi_price": None,
                "kalshi_prices": norm,
                "tickers": [m.get("ticker", "") for m in ms],
            })

    records.sort(key=lambda e: e["close_dt"])
    return records


# ── Prediction ────────────────────────────────────────────────────────────────

async def predict_all(events: list[dict], base_url: str) -> list[dict]:
    results = []
    async with httpx.AsyncClient() as client:
        for i, ev in enumerate(events, 1):
            close_dt = datetime.fromisoformat(ev["close_dt"])
            mins = (close_dt - datetime.now(timezone.utc)).total_seconds() / 60
            print(f"[{i}/{len(events)}] {ev['title'][:65]}")
            print(f"  Category  : {ev['category']}  |  Closes in: {mins:.0f} min")

            if ev["is_binary"]:
                print(f"  Kalshi    : {ev['kalshi_price']:.1%} YES")
            else:
                top = sorted((ev.get("kalshi_prices") or {}).items(), key=lambda x: -x[1])[:3]
                for name, p in top:
                    print(f"  Kalshi    : {name[:45]:<45} {p:.1%}")

            payload = {k: ev[k] for k in
                       ["event_ticker", "market_ticker", "title", "category", "rules", "close_time", "outcomes"]}
            try:
                r = await client.post(f"{base_url}/predict", json=payload, timeout=120.0)
                r.raise_for_status()
                prediction = r.json()
                # Echo our prediction
                if "p_yes" in prediction:
                    print(f"  Our p_yes : {prediction['p_yes']:.1%}")
                elif "probabilities" in prediction:
                    top_p = sorted(prediction["probabilities"], key=lambda x: -x["probability"])[:3]
                    for item in top_p:
                        print(f"  Our top   : {item['market'][:45]:<45} {item['probability']:.1%}")
            except Exception as e:
                print(f"  ERROR     : {e}")
                prediction = None

            results.append({
                **ev,
                "our_prediction": prediction,
                "predicted_at": datetime.now(timezone.utc).isoformat(),
            })
            print()

    return results


# ── Schedule score_live.py ────────────────────────────────────────────────────

def schedule_scoring(predictions_file: str, delay_seconds: int):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_live.py")
    log = predictions_file.replace("predictions_", "score_log_").replace(".json", ".log")
    cmd = f'nohup bash -c "sleep {delay_seconds} && {sys.executable} {script} --file {predictions_file}" > {log} 2>&1 &'
    subprocess.Popen(cmd, shell=True)
    run_at = datetime.now() + timedelta(seconds=delay_seconds)
    print(f"  score_live.py will run at : {run_at.strftime('%H:%M:%S')} (in {delay_seconds//60} min {delay_seconds%60} sec)")
    print(f"  Score log                 : {log}")


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(base_url: str, window_min: int, window_max: int):
    print(f"Searching for markets closing in {window_min}–{window_max} min...\n")
    events = await fetch_closing_soon(window_min, window_max)

    if not events:
        print("No markets found in that window. Try --window-min 30 --window-max 180.")
        return

    print(f"Found {len(events)} events:\n")
    for ev in events:
        mins = (datetime.fromisoformat(ev["close_dt"]) - datetime.now(timezone.utc)).total_seconds() / 60
        n = ev["n_markets"]
        print(f"  [{mins:4.0f} min]  {ev['title'][:60]}  ({n} market{'s' if n>1 else ''})")

    print(f"\nRunning predictions...\n{'─'*65}")
    predicted = await predict_all(events, base_url)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"predictions_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(predicted, f, indent=2, default=str)
    print(f"{'─'*65}")
    print(f"Predictions saved → {out_file}\n")

    # Schedule scoring: latest close + 10 min settlement buffer
    latest_close = max(datetime.fromisoformat(ev["close_dt"]) for ev in events)
    delay = max(int((latest_close - datetime.now(timezone.utc)).total_seconds()) + 600, 120)

    print("Scheduling score_live.py...")
    schedule_scoring(out_file, delay)
    print(f"\nAll done. Results will appear automatically in ~{delay//60} min.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--window-min", type=int, default=60)
    parser.add_argument("--window-max", type=int, default=120)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.window_min, args.window_max))
