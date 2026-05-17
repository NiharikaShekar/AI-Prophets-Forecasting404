import asyncio

from .calibration import calibrate_binary, calibrate_multi
from .event_classifier import EventType, classify
from .evidence.kalshi import get_binary_prior, get_multi_prior
from .evidence.search import (
    search_category,
    search_numeric,
    search_recent_news,
    search_stats_history,
)
from .reasoning.ensemble import run_ensemble


async def predict(event: dict) -> dict:
    classified = classify(event)
    ticker = event.get("market_ticker", "")
    category = event.get("category", "")
    close_time = event.get("close_time", "")
    outcomes = classified.outcomes
    title = event.get("title", "")

    # ── Stage 1: Parallel evidence gathering ─────────────────────────────
    normalize_prior = not classified.is_multi_label

    if classified.type == EventType.NUMERIC:
        kalshi_prior, news, stats, category_info = await asyncio.gather(
            get_multi_prior(ticker, outcomes, normalize=normalize_prior),
            search_recent_news(title),
            search_stats_history(title),
            search_numeric(title),
        )
    elif classified.type == EventType.BINARY:
        kalshi_prior, news, stats, category_info = await asyncio.gather(
            get_binary_prior(ticker),
            search_recent_news(title),
            search_stats_history(title),
            search_category(title, category),
        )
    else:  # MULTI
        kalshi_prior, news, stats, category_info = await asyncio.gather(
            get_multi_prior(ticker, outcomes, normalize=normalize_prior),
            search_recent_news(title),
            search_stats_history(title),
            search_category(title, category),
        )

    # ── Stage 2+3+4: Dual model reasoning ────────────────────────────────
    result = await run_ensemble(
        event=event,
        kalshi_prior=kalshi_prior,
        news=news,
        stats=stats,
        category_info=category_info,
        event_type=classified.type.value,
        multi_label=classified.is_multi_label,
    )

    evidence_quality = result.get("evidence_quality", "Moderate")
    rationale = result.get("rationale", "")

    # ── Stage 5: Calibration ─────────────────────────────────────────────
    if classified.type == EventType.BINARY:
        p_raw = float(result.get("p_yes", 0.5))
        prior = kalshi_prior if isinstance(kalshi_prior, float) else None
        p_final = calibrate_binary(p_raw, prior, close_time, evidence_quality)
        return {"p_yes": round(p_final, 4), "rationale": rationale}
    else:
        probs_raw = result.get("probabilities", {o: 1 / len(outcomes) for o in outcomes})
        prior_dict = kalshi_prior if isinstance(kalshi_prior, dict) else None
        probs_final = calibrate_multi(
            probs_raw,
            prior_dict,
            close_time,
            evidence_quality,
            outcomes,
            multi_label=classified.is_multi_label,
        )
        return {
            "probabilities": [
                {"market": o, "probability": round(probs_final[o], 4)}
                for o in outcomes
            ],
            "rationale": rationale,
        }
