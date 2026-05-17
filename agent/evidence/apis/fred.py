"""FRED (Federal Reserve Economic Data) — for Economics, Commodities, Financials."""
import logging
import os

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred"
_TIMEOUT = httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=2.0)

_SERIES_MAP = [
    (["inflation", "cpi", "consumer price"], "CPIAUCSL"),
    (["core inflation", "core cpi", "pce"], "PCEPILFE"),
    (["unemployment", "jobless", "jobs report", "nonfarm payroll"], "UNRATE"),
    (["gdp", "gross domestic product", "economic growth"], "GDP"),
    (["federal funds", "fed rate", "interest rate", "fomc", "rate cut", "rate hike", " fed "], "FEDFUNDS"),
    (["10 year", "10-year", "treasury yield", "bond yield"], "DGS10"),
    (["2 year", "2-year", "short term rate"], "DGS2"),
    (["oil", "crude", "wti", "brent", "petroleum"], "DCOILWTICO"),
    (["natural gas", "gas price", "henry hub"], "MHHNGSP"),
    (["gold", "gold price"], "GOLDAMGBD228NLBM"),
    (["copper"], "PCOPPUSDM"),
    (["silver"], "PSILVERUSDM"),
    (["dollar", "usd index", "dxy"], "DTWEXBGS"),
    (["s&p 500", "s&p500", "stock market", "equity"], "SP500"),
    (["nasdaq"], "NASDAQCOM"),
    (["housing", "home price", "case shiller"], "CSUSHPINSA"),
    (["retail sales", "consumer spending"], "RSXFS"),
    (["m2", "money supply"], "M2SL"),
    (["vix", "volatility"], "VIXCLS"),
]


def _detect_series(title: str) -> list[str]:
    t = title.lower()
    found = []
    for keywords, series_id in _SERIES_MAP:
        if any(kw in t for kw in keywords):
            found.append(series_id)
            if len(found) >= 2:
                break
    return found or ["CPIAUCSL", "UNRATE"]


async def _fetch_series(client: httpx.AsyncClient, series_id: str, api_key: str) -> str:
    try:
        r = await client.get(
            f"{_BASE}/series/observations",
            params={"series_id": series_id, "api_key": api_key,
                    "file_type": "json", "limit": 6, "sort_order": "desc"},
        )
        if r.status_code == 400:
            logger.debug("FRED: unknown series %s", series_id)
            return ""
        if r.status_code == 429:
            logger.warning("FRED: rate-limited")
            return ""
        if r.status_code != 200:
            logger.warning("FRED returned HTTP %d for series %s", r.status_code, series_id)
            return ""
        obs = r.json().get("observations", [])
        if not obs:
            return ""

        r2 = await client.get(
            f"{_BASE}/series",
            params={"series_id": series_id, "api_key": api_key, "file_type": "json"},
        )
        series_name = series_id
        if r2.status_code == 200:
            series_list = r2.json().get("seriess", [])
            if series_list:
                series_name = series_list[0].get("title", series_id)

        lines = [f"FRED — {series_name} ({series_id}):"]
        for o in obs[:6]:
            val = o.get("value", ".")
            if val != ".":
                lines.append(f"  {o.get('date', '')}: {val}")
        return "\n".join(lines) if len(lines) > 1 else ""

    except httpx.TimeoutException:
        logger.warning("FRED: timeout fetching series %s", series_id)
        return ""
    except Exception as exc:
        logger.warning("FRED error for series %s: %s", series_id, exc)
        return ""


async def get_fred_data(title: str) -> str:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        return ""
    if not title.strip():
        return ""
    series_ids = _detect_series(title)
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            results = []
            for sid in series_ids:
                res = await _fetch_series(client, sid, api_key)
                if res:
                    results.append(res)
            return "\n\n".join(results)
    except httpx.ConnectError:
        logger.warning("FRED: cannot connect")
        return ""
    except Exception as exc:
        logger.warning("FRED unexpected error: %s", exc)
        return ""
