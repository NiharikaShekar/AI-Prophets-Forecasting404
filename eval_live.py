"""
Live evaluation: predict on Kalshi events closing in 60-120 min, wait for
resolution, then compute real Brier scores against actual outcomes.

Usage:
    python eval_live.py [--url http://localhost:8000] [--window-min 60] [--window-max 120]
"""

import argparse
import asyncio
import json
from datetime import datetime, timezone, timedelta

import httpx

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
POLL_INTERVAL = 60   # seconds between result-check polls
MAX_WAIT = 7200      # give up after 2 hours past close time


# ── Kalshi helpers ────────────────────────────────────────────────────────────

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


def _parse_close(close_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(close_str.replace("Z", "+00:00"))
    except Exception:
        return None


async def fetch_closing_soon(window_min: int, window_max: int) -> list[dict]:
    """Return individual binary Kalshi markets closing in [window_min, window_max] minutes."""
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
                # Skip multivariate parlay markets
                if "MULTIVARIATE" in m.get("ticker", "").upper() or "MVE" in m.get("ticker", "").upper():
                    continue
                close_dt = _parse_close(m.get("close_time", ""))
                if close_dt and lo <= close_dt <= hi:
                    et = m.get("event_ticker", m.get("ticker", ""))
                    events.setdefault(et, []).append(m)

            cursor = body.get("cursor")
            if not cursor:
                break

    # Build event records
    result = []
    for event_ticker, ms in events.items():
        ms.sort(key=lambda m: m.get("ticker", ""))
        first = ms[0]
        close_dt = _parse_close(first.get("close_time", ""))

        if len(ms) == 1:
            price = _mid(first)
            result.append({
                "event_ticker": event_ticker,
                "market_ticker": first.get("ticker", event_ticker),
                "title": first.get("title", event_ticker),
                "category": first.get("category") or "General",
                "rules": first.get("rules_primary", ""),
                "close_time": first.get("close_time", ""),
                "close_dt": close_dt,
                "outcomes": ["Yes", "No"],
                "is_binary": True,
                "n_markets": 1,
                "kalshi_price": price,
                "tickers": [first.get("ticker", "")],
            })
        elif len(ms) >= 2:
            subtitles = [m.get("subtitle") or m.get("title", f"Option {i+1}") for i, m in enumerate(ms)]
            prices = {sub: _mid(m) for sub, m in zip(subtitles, ms)}
            total = sum(prices.values())
            norm = {k: round(v / total, 4) for k, v in prices.items()} if total > 0 else {}
            result.append({
                "event_ticker": event_ticker,
                "market_ticker": event_ticker,
                "title": first.get("title", event_ticker),
                "category": first.get("category") or "General",
                "rules": first.get("rules_primary", ""),
                "close_time": first.get("close_time", ""),
                "close_dt": close_dt,
                "outcomes": subtitles,
                "is_binary": False,
                "n_markets": len(ms),
                "kalshi_prices": norm,
                "tickers": [m.get("ticker", "") for m in ms],
            })

    result.sort(key=lambda e: e["close_dt"] or datetime.max.replace(tzinfo=timezone.utc))
    return result


async def fetch_result(ticker: str) -> str | None:
    """Poll a single market ticker and return 'yes'/'no' when settled, else None."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{KALSHI_BASE}/markets/{ticker}")
        if r.status_code != 200:
            return None
        m = r.json().get("market", r.json())
        result = m.get("result", "")
        status = m.get("status", "")
        if result in ("yes", "no"):
            return result
        if status in ("settled", "resolved"):
            return result or None
    return None


async def wait_for_results(event: dict) -> str | None:
    """Wait until the market settles and return 'yes'/'no'."""
    close_dt = event["close_dt"]
    now = datetime.now(timezone.utc)
    wait_secs = max(0, (close_dt - now).total_seconds()) + 30
    print(f"  Waiting {wait_secs/60:.1f} min for {event['title'][:55]} to close...")
    await asyncio.sleep(wait_secs)

    primary_ticker = event["tickers"][0]
    deadline = datetime.now(timezone.utc) + timedelta(seconds=MAX_WAIT)
    while datetime.now(timezone.utc) < deadline:
        result = await fetch_result(primary_ticker)
        if result:
            return result
        await asyncio.sleep(POLL_INTERVAL)
    return None


# ── Scoring ───────────────────────────────────────────────────────────────────

def brier_binary(p_yes: float, result: str) -> float:
    actual = 1.0 if result == "yes" else 0.0
    return (p_yes - actual) ** 2


def brier_multi(probs: list[dict], result: str, winning_subtitle: str) -> float:
    total = sum(
        (item["probability"] - (1.0 if item["market"] == winning_subtitle else 0.0)) ** 2
        for item in probs
    )
    return total / len(probs)


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_predictions(events: list[dict], base_url: str) -> list[dict]:
    results = []
    async with httpx.AsyncClient() as client:
        for i, ev in enumerate(events, 1):
            payload = {
                "event_ticker": ev["event_ticker"],
                "market_ticker": ev["market_ticker"],
                "title": ev["title"],
                "category": ev["category"],
                "rules": ev["rules"],
                "close_time": ev["close_time"],
                "outcomes": ev["outcomes"],
            }
            print(f"[{i}/{len(events)}] Predicting: {ev['title'][:60]}")
            if ev["is_binary"]:
                print(f"  Kalshi price: {ev.get('kalshi_price', 0):.1%} YES  | closes: {ev['close_time']}")
            else:
                top = sorted(ev.get("kalshi_prices", {}).items(), key=lambda x: -x[1])[:2]
                print(f"  Kalshi top:   {top}  | closes: {ev['close_time']}")
            try:
                r = await client.post(f"{base_url}/predict", json=payload, timeout=120.0)
                r.raise_for_status()
                prediction = r.json()
                print(f"  Our prediction: {json.dumps({k: v for k, v in prediction.items() if k != 'rationale'})}")
            except Exception as e:
                print(f"  ERROR: {e}")
                prediction = None
            results.append({**ev, "prediction": prediction})
            print()
    return results


async def score_results(predicted_events: list[dict]) -> list[dict]:
    scored = []
    for ev in predicted_events:
        if ev["prediction"] is None:
            print(f"SKIP (no prediction): {ev['title'][:60]}")
            continue

        print(f"Fetching result: {ev['title'][:60]}")
        result = await wait_for_results(ev)

        if result is None:
            print(f"  TIMEOUT — could not get result")
            continue

        print(f"  Settled: {result}")

        pred = ev["prediction"]
        if ev["is_binary"]:
            p_yes = pred.get("p_yes", 0.5)
            bs = brier_binary(p_yes, result)
            kalshi_bs = brier_binary(ev.get("kalshi_price", 0.5), result)
            correct = (p_yes >= 0.5 and result == "yes") or (p_yes < 0.5 and result == "no")
            print(f"  p_yes={p_yes:.3f} | result={result} | brier={bs:.4f} | kalshi_brier={kalshi_bs:.4f} | {'CORRECT' if correct else 'WRONG'}")
        else:
            probs = pred.get("probabilities", [])
            if not probs:
                continue
            top = max(probs, key=lambda x: x["probability"])
            winning_subtitle = "yes" if result == "yes" else "no"
            bs = brier_multi(probs, result, winning_subtitle)
            kalshi_bs = 0.25
            correct = None
            print(f"  top={top['market'][:40]}({top['probability']:.3f}) | result={result}")

        scored.append({
            **ev,
            "result": result,
            "our_brier": bs,
            "kalshi_brier": kalshi_bs if ev["is_binary"] else None,
            "correct": correct,
        })

    return scored


def print_summary(scored: list[dict]):
    if not scored:
        print("No scored events.")
        return

    our_avg = sum(s["our_brier"] for s in scored) / len(scored)
    binary = [s for s in scored if s["is_binary"] and s.get("kalshi_brier") is not None]

    print("\n" + "=" * 65)
    print("LIVE EVALUATION RESULTS")
    print("=" * 65)
    print(f"Events scored   : {len(scored)}")
    print(f"Our Brier       : {our_avg:.4f}")
    if binary:
        kb_avg = sum(s["kalshi_brier"] for s in binary) / len(binary)
        print(f"Kalshi Brier    : {kb_avg:.4f}  (crowd baseline)")
        print(f"vs Kalshi crowd : {'BETTER' if our_avg < kb_avg else 'WORSE'} by {abs(our_avg - kb_avg):.4f}")
    correct = [s for s in scored if s.get("correct") is True]
    wrong = [s for s in scored if s.get("correct") is False]
    print(f"Correct picks   : {len(correct)}/{len(correct)+len(wrong)}")
    print()
    for s in scored:
        marker = "✓" if s.get("correct") else "✗" if s.get("correct") is False else "?"
        print(f"  {marker} [{s['our_brier']:.3f}] {s['title'][:55]}")
    print("=" * 65)


async def main(base_url: str, window_min: int, window_max: int):
    print(f"Searching for Kalshi markets closing in {window_min}–{window_max} min...")
    events = await fetch_closing_soon(window_min, window_max)

    if not events:
        print(f"No markets found closing in {window_min}–{window_max} min. Try widening the window.")
        return

    print(f"Found {len(events)} events:\n")
    for ev in events:
        mins = (ev["close_dt"] - datetime.now(timezone.utc)).total_seconds() / 60
        print(f"  [{mins:.0f}min] {ev['title'][:60]}  ({ev['n_markets']} market{'s' if ev['n_markets']>1 else ''})")

    print(f"\nRunning predictions on {len(events)} events...\n")
    predicted = await run_predictions(events, base_url)

    # Save predictions immediately
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"live_predictions_{ts}.json"
    with open(save_path, "w") as f:
        json.dump(predicted, f, indent=2, default=str)
    print(f"Predictions saved → {save_path}\n")

    print("Now waiting for markets to close and settle...\n")
    scored = await score_results(predicted)

    if scored:
        result_path = f"live_results_{ts}.json"
        with open(result_path, "w") as f:
            json.dump(scored, f, indent=2, default=str)
        print(f"Results saved → {result_path}")

    print_summary(scored)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--window-min", type=int, default=60)
    parser.add_argument("--window-max", type=int, default=120)
    args = parser.parse_args()
    asyncio.run(main(args.url, args.window_min, args.window_max))
