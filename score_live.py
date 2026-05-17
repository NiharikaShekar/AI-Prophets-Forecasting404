"""
Script 2 of 2 — Score predictions against actual Kalshi outcomes.

Auto-scheduled by predict_live.py, or run manually:
    python score_live.py --file predictions_TIMESTAMP.json
"""

import argparse
import asyncio
import json

import httpx

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
MAX_RETRIES = 12
RETRY_INTERVAL = 60  # seconds


# ── Result fetching ───────────────────────────────────────────────────────────

async def fetch_result(ticker: str) -> str | None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(f"{KALSHI_BASE}/markets/{ticker}")
        if r.status_code != 200:
            return None
        body = r.json()
        market = body.get("market", body)
        result = market.get("result", "")
        return result if result in ("yes", "no") else None


async def fetch_result_with_retry(ticker: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        result = await fetch_result(ticker)
        if result:
            return result
        if attempt < MAX_RETRIES:
            print(f"    attempt {attempt}/{MAX_RETRIES}: not settled yet, retrying in {RETRY_INTERVAL}s...")
            await asyncio.sleep(RETRY_INTERVAL)
    return None


# ── Scoring ───────────────────────────────────────────────────────────────────

def brier_binary(p: float, result: str) -> float:
    actual = 1.0 if result == "yes" else 0.0
    return round((p - actual) ** 2, 6)


def brier_multi(probs: list[dict], winner: str) -> float:
    bs = sum(
        (item["probability"] - (1.0 if item["market"] == winner else 0.0)) ** 2
        for item in probs
    )
    return round(bs / len(probs), 6)


def correct_binary(p: float, result: str) -> bool:
    return (p >= 0.5 and result == "yes") or (p < 0.5 and result == "no")


# ── Per-event processing ──────────────────────────────────────────────────────

async def score_event(ev: dict) -> dict | None:
    title = ev.get("title", "?")
    prediction = ev.get("our_prediction")
    is_binary = ev.get("is_binary", True)
    primary_ticker = ev["tickers"][0] if ev.get("tickers") else ev.get("market_ticker", "")

    print(f"\nEvent  : {title[:70]}")
    print(f"Ticker : {primary_ticker}")
    print(f"Closed : {ev.get('close_time', '?')}")

    if prediction is None:
        print("  SKIP — agent errored during prediction")
        return None

    result = await fetch_result_with_retry(primary_ticker)
    if result is None:
        print("  SKIP — Kalshi result not available after max retries")
        return None

    print(f"  Actual result : {result.upper()}")

    if is_binary:
        our_p     = prediction.get("p_yes", 0.5)
        kalshi_p  = ev.get("kalshi_price", 0.5)

        our_bs    = brier_binary(our_p, result)
        kalshi_bs = brier_binary(kalshi_p, result)

        our_ok    = correct_binary(our_p, result)
        kalshi_ok = correct_binary(kalshi_p, result)

        print(f"  {'Source':<12} {'Prediction':>12}  {'Brier':>8}  {'Pick'}")
        print(f"  {'─'*50}")
        print(f"  {'Our agent':<12} {our_p:>11.1%}  {our_bs:>8.4f}  {'✓ CORRECT' if our_ok else '✗ WRONG'}")
        print(f"  {'Kalshi crowd':<12} {kalshi_p:>11.1%}  {kalshi_bs:>8.4f}  {'✓ CORRECT' if kalshi_ok else '✗ WRONG'}")

        return {
            "title": title, "is_binary": True, "result": result,
            "our_p": our_p, "kalshi_p": kalshi_p,
            "our_brier": our_bs, "kalshi_brier": kalshi_bs,
            "our_correct": our_ok, "kalshi_correct": kalshi_ok,
        }

    else:
        # Multi-outcome: find which subtitle won by checking each market ticker
        winner = None
        async with httpx.AsyncClient(timeout=10.0) as client:
            for ticker, subtitle in zip(ev.get("tickers", []), ev.get("outcomes", [])):
                r = await client.get(f"{KALSHI_BASE}/markets/{ticker}")
                if r.status_code == 200:
                    m = r.json().get("market", r.json())
                    if m.get("result") == "yes":
                        winner = subtitle
                        break

        probs = prediction.get("probabilities", [])
        if not probs:
            print("  SKIP — no probabilities in prediction")
            return None

        our_bs   = brier_multi(probs, winner or "")
        top_ours = max(probs, key=lambda x: x["probability"])
        our_ok   = top_ours["market"] == winner if winner else None

        kalshi_prices = ev.get("kalshi_prices") or {}
        top_kalshi = max(kalshi_prices, key=kalshi_prices.get) if kalshi_prices else None
        kalshi_ok  = top_kalshi == winner if (top_kalshi and winner) else None

        print(f"  Winner     : {winner or '(could not determine)'}")
        print(f"  Our top    : {top_ours['market'][:50]} ({top_ours['probability']:.1%})  {'✓' if our_ok else '✗' if our_ok is False else '?'}")
        if top_kalshi:
            print(f"  Kalshi top : {top_kalshi[:50]} ({kalshi_prices[top_kalshi]:.1%})  {'✓' if kalshi_ok else '✗' if kalshi_ok is False else '?'}")
        print(f"  Our Brier  : {our_bs:.4f}")

        return {
            "title": title, "is_binary": False, "result": result, "winner": winner,
            "our_brier": our_bs, "kalshi_brier": None,
            "our_correct": our_ok, "kalshi_correct": kalshi_ok,
        }


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(scored: list[dict]):
    print("\n" + "=" * 70)
    print("FINAL RESULTS — OUR AGENT vs KALSHI CROWD")
    print("=" * 70)

    binary = [s for s in scored if s["is_binary"]]
    multi  = [s for s in scored if not s["is_binary"]]

    # Header
    print(f"\n{'Event':<40} {'Result':<6} {'OurB':>6} {'KalB':>6} {'OurPick':>8} {'KalPick':>8}")
    print("─" * 80)
    for s in scored:
        our_b  = f"{s['our_brier']:.4f}"
        kal_b  = f"{s['kalshi_brier']:.4f}" if s.get("kalshi_brier") is not None else "  n/a"
        our_ok = "✓" if s.get("our_correct") else "✗" if s.get("our_correct") is False else "?"
        kal_ok = "✓" if s.get("kalshi_correct") else "✗" if s.get("kalshi_correct") is False else "?"
        print(f"  {s['title'][:38]:<38} {s['result'].upper():<6} {our_b:>6} {kal_b:>6} {our_ok:>8} {kal_ok:>8}")

    print("─" * 80)

    # Aggregate metrics
    our_avg = sum(s["our_brier"] for s in scored) / len(scored)
    our_correct_n   = sum(1 for s in scored if s.get("our_correct") is True)
    kalshi_correct_n = sum(1 for s in scored if s.get("kalshi_correct") is True)
    total_pick = len([s for s in scored if s.get("our_correct") is not None])

    print(f"\n  Events scored       : {len(scored)}")
    print(f"  Our avg Brier       : {our_avg:.4f}  (lower is better, 0=perfect, 0.25=random)")

    if binary:
        kal_avg = sum(s["kalshi_brier"] for s in binary) / len(binary)
        our_bin = sum(s["our_brier"] for s in binary) / len(binary)
        diff    = our_bin - kal_avg
        verdict = "BEAT the crowd ✓" if diff < 0 else "LOST to the crowd ✗"
        print(f"  Kalshi avg Brier    : {kal_avg:.4f}  (binary events only)")
        print(f"  vs Kalshi crowd     : {verdict}  (diff {diff:+.4f})")

    if multi:
        print(f"  Multi-outcome Brier : {sum(s['our_brier'] for s in multi)/len(multi):.4f}")

    print(f"\n  Correct picks — us     : {our_correct_n}/{total_pick}")
    if binary:
        print(f"  Correct picks — Kalshi : {kalshi_correct_n}/{len(binary)}")

    print("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(predictions_file: str):
    print(f"Loading {predictions_file}...")
    with open(predictions_file) as f:
        events = json.load(f)
    print(f"Loaded {len(events)} events. Fetching Kalshi results...\n")

    scored = []
    for ev in events:
        row = await score_event(ev)
        if row:
            scored.append(row)

    if not scored:
        print("\nNo events could be scored.")
        return

    print_summary(scored)

    out = predictions_file.replace("predictions_", "scored_")
    with open(out, "w") as f:
        json.dump(scored, f, indent=2, default=str)
    print(f"\nFull results saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="predictions_TIMESTAMP.json from predict_live.py")
    args = parser.parse_args()
    asyncio.run(main(args.file))
