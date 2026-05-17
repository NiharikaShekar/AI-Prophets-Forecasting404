import httpx

KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"


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
    # Thin market: only ask is available, use it directly
    if 0.01 <= yes_ask <= 0.99:
        return yes_ask
    return 0.0


async def get_binary_prior(market_ticker: str) -> float | None:
    url = f"{KALSHI_BASE}/markets/{market_ticker}"
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            data = resp.json()
            market = data.get("market", data)
            price = _mid(market)
            return price if price > 0 else None
        except Exception:
            return None


async def get_multi_prior(event_ticker: str, outcomes: list[str]) -> dict[str, float] | None:
    url = f"{KALSHI_BASE}/markets"
    async with httpx.AsyncClient(timeout=8.0) as client:
        try:
            resp = await client.get(
                url,
                params={"event_ticker": event_ticker, "status": "open", "limit": 200},
            )
            if resp.status_code != 200:
                return None
            markets = resp.json().get("markets", [])
            if not markets:
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
                return None

            # Normalize — Kalshi multi-outcome markets don't sum to 1
            total = sum(result.values())
            return {k: v / total for k, v in result.items()}
        except Exception:
            return None
