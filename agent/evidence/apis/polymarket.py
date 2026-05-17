"""Polymarket Gamma API — prediction market data for Elections/Politics events."""
import json
import logging

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://gamma-api.polymarket.com"
_TIMEOUT = httpx.Timeout(connect=3.0, read=7.0, write=3.0, pool=2.0)


def _parse_outcome_prices(prices, outcomes) -> str:
    try:
        if isinstance(prices, str):
            prices = json.loads(prices)
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        pairs = list(zip(outcomes, prices))[:4]
        return "  |  ".join(f"{o}: {float(p):.1%}" for o, p in pairs)
    except Exception:
        return ""


async def get_polymarket_data(title: str) -> str:
    if not title.strip():
        return ""
    query = " ".join(
        w for w in title.split()
        if w.lower() not in {"will", "the", "a", "an", "in", "of", "to",
                             "by", "for", "be", "vs", "and"}
    )[:80]
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_BASE}/markets",
                params={"keyword": query, "active": "true", "limit": 5},
            )
            if r.status_code == 429:
                logger.warning("Polymarket: rate-limited")
                return ""
            if r.status_code != 200:
                logger.warning("Polymarket returned HTTP %d for query '%s'", r.status_code, query[:40])
                return ""
            markets = r.json()
            if not isinstance(markets, list) or not markets:
                logger.debug("Polymarket: no markets found for '%s'", query[:40])
                return ""

            lines = ["Polymarket prediction markets (similar questions):"]
            for m in markets[:4]:
                q = m.get("question", "")
                volume = float(m.get("volume") or 0)
                prob_str = _parse_outcome_prices(m.get("outcomePrices"), m.get("outcomes"))
                lines.append(f"  Q: {q[:80]}")
                lines.append(f"     {prob_str}  (volume: ${volume:,.0f})" if prob_str else f"     no odds  (volume: ${volume:,.0f})")
            return "\n".join(lines)

    except httpx.TimeoutException:
        logger.warning("Polymarket: timeout for query '%s'", query[:40])
        return ""
    except httpx.ConnectError:
        logger.warning("Polymarket: cannot connect")
        return ""
    except Exception as exc:
        logger.warning("Polymarket unexpected error: %s", exc)
        return ""
