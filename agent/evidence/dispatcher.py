"""
Evidence dispatcher: classify event category → call Tavily + domain APIs in parallel.

Tavily is always called regardless of category. Category-specific APIs are called
on top and their results are appended to the evidence string.

If the category is missing or unrecognized, keyword-based auto-classification runs
as a fallback using the event title and rules text.
"""
import asyncio
import logging
from typing import Callable, Coroutine, Any

from .search import _search, _format, _CATEGORY_QUERIES
from .apis.odds import get_sports_odds
from .apis.coingecko import get_crypto_data
from .apis.fred import get_fred_data
from .apis.polymarket import get_polymarket_data
from .apis.reddit import get_reddit_mentions
from .apis.noaa import get_climate_data
from .apis.tmdb import get_tmdb_data
from .apis.edgar import get_edgar_filings

logger = logging.getLogger(__name__)

_CATEGORY_APIS: dict[str, list[Callable[[str], Coroutine[Any, Any, str]]]] = {
    "Sports":         [get_sports_odds],
    "Elections":      [get_polymarket_data],
    "Politics":       [get_polymarket_data],
    "Crypto":         [get_crypto_data],
    "Commodities":    [get_fred_data],
    "Economics":      [get_fred_data],
    "Financials":     [get_fred_data, get_edgar_filings],
    "Mentions":       [get_reddit_mentions],
    "Climate":        [get_climate_data],
    "Culture":        [get_tmdb_data, get_reddit_mentions],
    "Companies":      [get_edgar_filings, get_fred_data],
    "Tech & Science": [],
}

_KEYWORD_CATEGORY: list[tuple[list[str], str]] = [
    (["ufc", "nfl", "nba", "mlb", "nhl", "fifa", "tennis", "golf", "mma", "fight",
      "match", "game", "tournament", "championship", "league", "season", "playoff",
      "world cup", "super bowl", "stanley cup", "olympics", "athlete", "team"], "Sports"),
    (["bitcoin", "btc", "ethereum", "eth", "crypto", "coin", "token", "defi",
      "blockchain", "solana", "altcoin", "nft", "web3"], "Crypto"),
    (["election", "vote", "ballot", "candidate", "polling", "senate", "house",
      "governor", "president", "congress", "primary", "referendum"], "Elections"),
    (["bill", "legislation", "policy", "government", "administration", "minister",
      "geopolitical", "sanction", "treaty", "nato", "diplomatic"], "Politics"),
    (["cpi", "inflation", "gdp", "unemployment", "fed rate", "interest rate",
      "rate cut", "rate hike", "recession", "economic growth", "fomc",
      "federal reserve", " fed ", "tariff", "trade deficit"], "Economics"),
    (["oil", "gold", "copper", "silver", "wheat", "corn", "natural gas",
      "commodity", "crude", "energy", "brent", "wti"], "Commodities"),
    (["stock", "earnings", "revenue", "profit", "ipo", "nasdaq", "s&p", "dow",
      "market cap", "dividend", "analyst", "price target", "sec filing"], "Financials"),
    (["tesla", "apple", "google", "microsoft", "amazon", "nvidia", "meta",
      "quarterly", "q1", "q2", "q3", "q4", "earnings", "ceo", "acquisition"], "Companies"),
    (["hurricane", "tornado", "earthquake", "flood", "temperature", "rainfall",
      "snowfall", "wildfire", "climate", "weather", "storm", "drought"], "Climate"),
    (["oscar", "emmy", "grammy", "award", "film", "movie", "album", "song",
      "box office", "streaming", "netflix", "spotify", "concert", "celebrity"], "Culture"),
    (["reddit", "twitter", "tiktok", "instagram", "youtube", "viral", "trend",
      "hashtag", "followers", "views", "mention", "social media"], "Mentions"),
    (["ai ", "artificial intelligence", "model release", "gpt", "llm", "study",
      "research", "paper", "discovery", "launch", "patent", "quantum"], "Tech & Science"),
]


def _classify_by_keywords(title: str, rules: str = "") -> str:
    text = (title + " " + rules).lower()
    scores: dict[str, int] = {}
    for keywords, cat in _KEYWORD_CATEGORY:
        hits = sum(1 for kw in keywords if kw in text)
        if hits:
            scores[cat] = scores.get(cat, 0) + hits
    if not scores:
        return "Tech & Science"
    return max(scores, key=lambda c: scores[c])


async def _tavily_for_category(title: str, category: str) -> str:
    template = _CATEGORY_QUERIES.get(category)
    query = template.replace("{title}", title) if template else f"{title} prediction forecast outcome analysis"
    results = await _search(query, max_results=6)
    return _format(results)


async def _safe_call(fn: Callable, title: str, name: str) -> str:
    """Call an API function and catch any exception that slipped through its own handler."""
    try:
        result = await fn(title)
        return result or ""
    except Exception as exc:
        logger.warning("Evidence API '%s' raised uncaught exception: %s", name, exc)
        return ""


async def gather_evidence(title: str, category: str, rules: str = "") -> str:
    """
    Gather all evidence for a given event title + category.
    Returns a single formatted string: Tavily results + domain API results.

    Falls back to keyword classification if category is empty or unrecognized.
    Always returns a non-empty string (at minimum "No evidence gathered.").
    """
    if not title.strip():
        logger.warning("gather_evidence called with empty title")
        return "No title provided."

    if category not in _CATEGORY_APIS:
        inferred = _classify_by_keywords(title, rules)
        logger.info("Category '%s' unrecognized — inferred '%s' for '%s'", category, inferred, title[:50])
        category = inferred

    domain_fns = _CATEGORY_APIS.get(category, [])
    api_names = ["tavily"] + [fn.__name__ for fn in domain_fns]

    coros: list[Coroutine] = [_tavily_for_category(title, category)]
    coros.extend(_safe_call(fn, title, fn.__name__) for fn in domain_fns)

    raw_results = await asyncio.gather(*coros, return_exceptions=True)

    parts: list[str] = []
    for name, r in zip(api_names, raw_results):
        if isinstance(r, Exception):
            logger.warning("Evidence API '%s' raised in gather: %s", name, r)
        elif isinstance(r, str) and r.strip():
            parts.append(r.strip())
        else:
            logger.debug("Evidence API '%s' returned empty for '%s'", name, title[:50])

    if not parts:
        logger.warning("All evidence APIs returned empty for '%s' (category: %s)", title[:50], category)
        return "No evidence gathered."

    logger.info("Evidence gathered for '%s': %d source(s) — %s",
                title[:50], len(parts), ", ".join(api_names[:len(parts)]))
    return "\n\n─────────────────────────\n\n".join(parts)
