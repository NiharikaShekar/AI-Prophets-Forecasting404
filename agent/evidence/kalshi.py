import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
_TIMEOUT = httpx.Timeout(connect=4.0, read=10.0, write=4.0, pool=2.0)


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
        mid = (yes_ask + yes_bid) / 2
        if 0.01 <= mid <= 0.99:
            return mid
    last = _parse_price(market.get("last_price_dollars") or market.get("last_price"))
    if 0.01 <= last <= 0.99:
        return last
    if 0.01 <= yes_ask <= 0.99:
        return yes_ask
    return 0.0


async def get_binary_prior(market_ticker: str) -> float | None:
    if not market_ticker:
        return None
    url = f"{KALSHI_BASE}/markets/{market_ticker}"
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url)
                if resp.status_code == 429:
                    logger.warning("Kalshi rate-limited fetching binary prior for %s", market_ticker)
                    return None
                if resp.status_code != 200:
                    logger.warning("Kalshi binary prior HTTP %d for %s", resp.status_code, market_ticker)
                    return None
                data = resp.json()
                market = data.get("market", data)
                price = _mid(market)
                return price if price > 0 else None
        except httpx.TimeoutException:
            if attempt == 0:
                logger.warning("Kalshi timeout for %s — retrying", market_ticker)
                await asyncio.sleep(1)
                continue
            logger.warning("Kalshi timeout for %s after retry", market_ticker)
            return None
        except httpx.ConnectError:
            logger.warning("Cannot connect to Kalshi for %s", market_ticker)
            return None
        except Exception as exc:
            logger.warning("Kalshi binary prior unexpected error for %s: %s", market_ticker, exc)
            return None
    return None


async def get_multi_prior(
    event_ticker: str,
    outcomes: list[str],
    *,
    normalize: bool = True,
) -> dict[str, float] | None:
    if not event_ticker or not outcomes:
        return None
    url = f"{KALSHI_BASE}/markets"
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(
                    url,
                    params={"event_ticker": event_ticker, "status": "open", "limit": 200},
                )
                if resp.status_code == 429:
                    logger.warning("Kalshi rate-limited fetching multi prior for %s", event_ticker)
                    return None
                if resp.status_code != 200:
                    logger.warning("Kalshi multi prior HTTP %d for %s", resp.status_code, event_ticker)
                    return None
                markets = resp.json().get("markets", [])
                if not markets:
                    logger.debug("Kalshi returned no markets for %s", event_ticker)
                    return None

                result: dict[str, float] = {}
                for market in markets:
                    subtitle = (market.get("subtitle") or market.get("title") or "").lower()
                    price = _mid(market)
                    if price <= 0:
                        continue
                    for outcome in outcomes:
                        if outcome.lower() in subtitle or subtitle in outcome.lower():
                            result[outcome] = price
                            break

                if len(result) < 2:
                    logger.debug("Kalshi matched only %d/%d outcomes for %s", len(result), len(outcomes), event_ticker)
                    return None

                if not normalize:
                    return result

                total = sum(result.values())
                if total <= 0:
                    return None
                return {k: v / total for k, v in result.items()}

        except httpx.TimeoutException:
            if attempt == 0:
                logger.warning("Kalshi timeout for %s — retrying", event_ticker)
                await asyncio.sleep(1)
                continue
            logger.warning("Kalshi timeout for %s after retry", event_ticker)
            return None
        except httpx.ConnectError:
            logger.warning("Cannot connect to Kalshi for %s", event_ticker)
            return None
        except Exception as exc:
            logger.warning("Kalshi multi prior unexpected error for %s: %s", event_ticker, exc)
            return None
    return None
