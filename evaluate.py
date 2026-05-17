"""
Evaluate our agent against the ai-prophet sample-resolved dataset.
Runs all 26 events through POST /predict and computes Brier scores.

Usage:
    python evaluate.py [--url http://localhost:8000] [--concurrency 3]
"""

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone

import httpx

DATASET_URL = (
    "https://raw.githubusercontent.com/ai-prophet/ai-prophet-datasets"
    "/main/datasets/sample-resolved/releases/v1.0.0/tasks.jsonl"
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def brier_binary(p_yes: float, resolved: list[str], outcomes: list[str]) -> float:
    actual = 1.0 if resolved and resolved[0] == outcomes[0] else 0.0
    return (p_yes - actual) ** 2


def brier_multi(probs: list[dict], resolved: list[str]) -> float:
    resolved_set = set(resolved)
    total = sum((item["probability"] - (1.0 if item["market"] in resolved_set else 0.0)) ** 2
                for item in probs)
    return total / len(probs)


def baseline_brier_binary() -> float:
    return 0.25  # p_yes=0.5 always → (0.5-0)^2 or (0.5-1)^2 = 0.25


def baseline_brier_multi(n: int) -> float:
    p = 1 / n
    return ((1 - p) ** 2 + (n - 1) * p ** 2) / n


def build_request(task: dict) -> dict:
    src = task["metadata"]["source"]
    return {
        "event_ticker": src.get("event_ticker", task["task_id"]),
        "market_ticker": task["task_id"],
        "title": task["title"],
        "description": task.get("context", ""),
        "category": task["metadata"]["category"],
        "rules": src.get("rules", task.get("context", "")),
        "close_time": src.get("close_time", task.get("predict_by", "")),
        "outcomes": task["outcomes"],
    }


# ── Core evaluation ──────────────────────────────────────────────────────────

async def evaluate_one(
    client: httpx.AsyncClient,
    base_url: str,
    task: dict,
    sem: asyncio.Semaphore,
    idx: int,
    total: int,
) -> dict:
    outcomes = task["outcomes"]
    resolved = task["resolved_outcome"]["value"]
    is_binary = len(outcomes) <= 2

    async with sem:
        print(f"[{idx}/{total}] {task['title'][:70]}")
        try:
            resp = await client.post(
                f"{base_url}/predict",
                json=build_request(task),
                timeout=120.0,
            )
            resp.raise_for_status()
            prediction = resp.json()
        except Exception as e:
            print(f"  ERROR: {e}")
            prediction = None

    if prediction is None:
        our_brier = baseline_brier_binary() if is_binary else baseline_brier_multi(len(outcomes))
        status = "ERROR"
    elif is_binary:
        p_yes = prediction.get("p_yes", 0.5)
        our_brier = brier_binary(p_yes, resolved, outcomes)
        predicted_winner = outcomes[0] if p_yes >= 0.5 else outcomes[1]
        actual_winner = resolved[0] if resolved else "?"
        status = "CORRECT" if predicted_winner == actual_winner else "WRONG"
        print(f"  p_yes={p_yes:.3f} | predicted={predicted_winner} | actual={actual_winner} | {status}")
    else:
        probs = prediction.get("probabilities", [])
        our_brier = brier_multi(probs, resolved)
        if probs:
            top = max(probs, key=lambda x: x["probability"])
            actual_winner = resolved[0] if resolved else "?"
            status = "CORRECT" if top["market"] == actual_winner else "WRONG"
            print(f"  top={top['market']}({top['probability']:.3f}) | actual={actual_winner} | {status}")
        else:
            status = "ERROR"

    base_brier = baseline_brier_binary() if is_binary else baseline_brier_multi(len(outcomes))

    return {
        "task_id": task["task_id"],
        "title": task["title"],
        "category": task["metadata"]["category"],
        "is_binary": is_binary,
        "n_outcomes": len(outcomes),
        "resolved": resolved,
        "our_brier": our_brier,
        "baseline_brier": base_brier,
        "status": status,
        "prediction": prediction,
    }


async def main(base_url: str, concurrency: int):
    # Load dataset
    print("Fetching dataset...")
    async with httpx.AsyncClient() as client:
        resp = await client.get(DATASET_URL, timeout=30.0)
        tasks = [json.loads(line) for line in resp.text.strip().splitlines()]
    print(f"Loaded {len(tasks)} events\n")

    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(*[
            evaluate_one(client, base_url, task, sem, i + 1, len(tasks))
            for i, task in enumerate(tasks)
        ])

    # ── Summary ───────────────────────────────────────────────────────────────
    our_avg = sum(r["our_brier"] for r in results) / len(results)
    base_avg = sum(r["baseline_brier"] for r in results) / len(results)
    correct = sum(1 for r in results if r["status"] == "CORRECT")
    errors = sum(1 for r in results if r["status"] == "ERROR")

    binary_results = [r for r in results if r["is_binary"]]
    multi_results = [r for r in results if not r["is_binary"]]

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"Total events   : {len(results)}")
    print(f"Correct picks  : {correct}/{len(results) - errors}  ({100*correct/max(1,len(results)-errors):.1f}%)")
    print(f"Errors         : {errors}")
    print()
    print(f"Our Brier      : {our_avg:.4f}  (lower is better)")
    print(f"Baseline Brier : {base_avg:.4f}  (random guesser)")
    improvement = (base_avg - our_avg) / base_avg * 100
    print(f"Improvement    : {improvement:+.1f}%  vs baseline")
    print()

    if binary_results:
        b_our = sum(r["our_brier"] for r in binary_results) / len(binary_results)
        b_base = sum(r["baseline_brier"] for r in binary_results) / len(binary_results)
        print(f"Binary  ({len(binary_results):2d} events): ours={b_our:.4f}  baseline={b_base:.4f}")
    if multi_results:
        m_our = sum(r["our_brier"] for r in multi_results) / len(multi_results)
        m_base = sum(r["baseline_brier"] for r in multi_results) / len(multi_results)
        print(f"Multi   ({len(multi_results):2d} events): ours={m_our:.4f}  baseline={m_base:.4f}")

    # Per-category breakdown
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r["category"], []).append(r)
    print("\nBy category:")
    for cat, rs in sorted(cats.items()):
        avg = sum(r["our_brier"] for r in rs) / len(rs)
        print(f"  {cat:<20} {len(rs):2d} events  brier={avg:.4f}")

    print("=" * 65)

    # Save full results
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("Full results saved → eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Number of parallel requests (default 3)")
    args = parser.parse_args()
    asyncio.run(main(args.url, args.concurrency))
