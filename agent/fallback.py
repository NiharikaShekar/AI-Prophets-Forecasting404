"""Error-path predictions: Kalshi market price first, then neutral."""

from .event_classifier import EventType, classify
from .evidence.kalshi import get_binary_prior, get_multi_prior


def _binary_from_prior(prior: float | None) -> dict:
    if prior is not None:
        return {
            "p_yes": round(prior, 4),
            "rationale": "Internal error — using Kalshi market price as fallback.",
        }
    return {
        "p_yes": 0.5,
        "rationale": "Internal error — market price unavailable; using neutral 0.5.",
    }


def _multi_from_prior(
    prior_dict: dict[str, float] | None,
    outcomes: list[str],
    *,
    multi_label: bool = False,
) -> dict:
    n = len(outcomes)
    if not outcomes:
        return {"probabilities": [], "rationale": "Internal error — no outcomes provided."}

    if prior_dict:
        probs = {o: prior_dict.get(o, 1 / n) for o in outcomes}
        if not multi_label:
            total = sum(probs.values())
            if total > 0:
                probs = {o: v / total for o, v in probs.items()}
        return {
            "probabilities": [
                {"market": o, "probability": round(max(0.01, min(0.99, probs[o])), 4)}
                for o in outcomes
            ],
            "rationale": "Internal error — using Kalshi market prices as fallback.",
        }

    return {
        "probabilities": [
            {"market": o, "probability": round(1 / n, 4)} for o in outcomes
        ],
        "rationale": "Internal error — market prices unavailable; using uniform fallback.",
    }


async def market_fallback(event: dict) -> dict:
    """Fetch Kalshi prior; on failure use 0.5 (binary) or uniform 1/N (multi)."""
    classified = classify(event)
    ticker = event.get("market_ticker", "")
    outcomes = classified.outcomes
    multi_label = classified.is_multi_label

    if classified.type == EventType.BINARY:
        prior = await get_binary_prior(ticker)
        return _binary_from_prior(prior)

    prior_dict = await get_multi_prior(
        ticker, outcomes, normalize=not multi_label
    )
    return _multi_from_prior(prior_dict, outcomes, multi_label=multi_label)


def binary_from_kalshi_prior(kalshi_prior: float | dict | None) -> float:
    """Resolve binary ensemble failure: market mid, else 0.5."""
    if isinstance(kalshi_prior, float):
        return kalshi_prior
    return 0.5


def multi_from_kalshi_prior(
    kalshi_prior: float | dict | None,
    outcomes: list[str],
    multi_label: bool = False,
) -> dict[str, float]:
    """Resolve multi ensemble failure: market prices, else uniform."""
    n = len(outcomes)
    if isinstance(kalshi_prior, dict) and kalshi_prior:
        probs = {o: kalshi_prior.get(o, 1 / n) for o in outcomes}
        if not multi_label:
            total = sum(probs.values())
            if total > 0:
                return {o: v / total for o, v in probs.items()}
        return probs
    return {o: 1 / n for o in outcomes}
