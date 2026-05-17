"""CoinGecko + Alternative.me Fear & Greed Index — for Crypto category events."""
import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

_CG_BASE = "https://api.coingecko.com/api/v3"
_FNG_URL = "https://api.alternative.me/fng/?limit=30"
_TIMEOUT = httpx.Timeout(connect=3.0, read=7.0, write=3.0, pool=2.0)

_COIN_MAP = [
    (["bitcoin", "btc"], "bitcoin"),
    (["ethereum", "eth"], "ethereum"),
    (["solana", "sol"], "solana"),
    (["cardano", "ada"], "cardano"),
    (["xrp", "ripple"], "ripple"),
    (["dogecoin", "doge"], "dogecoin"),
    (["avalanche", "avax"], "avalanche-2"),
    (["polkadot", "dot"], "polkadot"),
    (["chainlink", "link"], "chainlink"),
    (["bnb", "binance coin"], "binancecoin"),
    (["shiba", "shib"], "shiba-inu"),
    (["litecoin", "ltc"], "litecoin"),
    (["polygon", "matic"], "matic-network"),
    (["pepe"], "pepe"),
    (["sui"], "sui"),
    (["toncoin", "ton"], "the-open-network"),
]


def _detect_coin(title: str) -> str | None:
    t = title.lower()
    for keywords, coin_id in _COIN_MAP:
        if any(kw in t for kw in keywords):
            return coin_id
    return None


async def _get_fear_greed() -> str:
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(_FNG_URL)
            if r.status_code != 200:
                logger.warning("Fear & Greed API returned HTTP %d", r.status_code)
                return ""
            data = r.json().get("data", [])
            if not data:
                return ""
            latest = data[0]
            value = latest.get("value", "?")
            label = latest.get("value_classification", "?")
            trend = [f"{d.get('value')} ({d.get('value_classification')})" for d in data[:7]]
            return (
                f"Crypto Fear & Greed Index (Alternative.me):\n"
                f"  Current: {value}/100 — {label}\n"
                f"  7-day trend: {' → '.join(reversed(trend))}"
            )
    except httpx.TimeoutException:
        logger.warning("Fear & Greed API: timeout")
        return ""
    except httpx.ConnectError:
        logger.warning("Fear & Greed API: cannot connect")
        return ""
    except Exception as exc:
        logger.warning("Fear & Greed API unexpected error: %s", exc)
        return ""


async def _get_coingecko(coin_id: str) -> str:
    api_key = os.environ.get("COINGECKO_API_KEY", "")
    headers = {"x-cg-demo-api-key": api_key} if api_key else {}
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT, headers=headers) as client:
            r = await client.get(
                f"{_CG_BASE}/coins/markets",
                params={
                    "vs_currency": "usd",
                    "ids": coin_id,
                    "price_change_percentage": "1d,7d,30d",
                    "sparkline": "false",
                },
            )
            if r.status_code == 429:
                logger.warning("CoinGecko: rate-limited for coin %s", coin_id)
                return ""
            if r.status_code != 200:
                logger.warning("CoinGecko returned HTTP %d for coin %s", r.status_code, coin_id)
                return ""
            coins = r.json()
            if not isinstance(coins, list) or not coins:
                logger.debug("CoinGecko: no data for coin %s", coin_id)
                return ""
            c = coins[0]
            name = c.get("name", coin_id)
            price = c.get("current_price") or 0
            change_1d = c.get("price_change_percentage_1d_in_currency") or 0
            change_7d = c.get("price_change_percentage_7d_in_currency") or 0
            change_30d = c.get("price_change_percentage_30d_in_currency") or 0
            ath = c.get("ath") or 0
            ath_pct = c.get("ath_change_percentage") or 0
            cap = c.get("market_cap") or 0
            return (
                f"CoinGecko market data ({name}):\n"
                f"  Price: ${price:,.2f}  |  24h: {change_1d:+.1f}%  |  7d: {change_7d:+.1f}%  |  30d: {change_30d:+.1f}%\n"
                f"  Market cap: ${cap:,.0f}  |  ATH: ${ath:,.2f} ({ath_pct:+.1f}% from ATH)"
            )
    except httpx.TimeoutException:
        logger.warning("CoinGecko: timeout for coin %s", coin_id)
        return ""
    except httpx.ConnectError:
        logger.warning("CoinGecko: cannot connect")
        return ""
    except Exception as exc:
        logger.warning("CoinGecko unexpected error for %s: %s", coin_id, exc)
        return ""


async def get_crypto_data(title: str) -> str:
    if not title.strip():
        return ""
    coin_id = _detect_coin(title)
    coros = [_get_fear_greed()]
    if coin_id:
        coros.append(_get_coingecko(coin_id))
    parts = await asyncio.gather(*coros)
    return "\n\n".join(p for p in parts if p)
