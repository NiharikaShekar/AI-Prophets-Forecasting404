import asyncio
import os

from tavily import AsyncTavilyClient

_CATEGORY_QUERIES: dict[str, str] = {
    "Elections": "{title} polling forecast prediction latest",
    "Politics": "{title} expert analysis vote forecast outcome",
    "Sports": "{title} injury report team form head to head stats",
    "Culture": "{title} prediction award odds expert picks",
    "Crypto": "{title} price prediction analyst forecast on-chain",
    "Commodities": "{title} price forecast analyst supply demand",
    "Climate": "{title} forecast model NOAA weather data prediction",
    "Economics": "{title} analyst forecast consensus estimate GDP",
    "Mentions": "{title} trending social media count data",
    "Companies": "{title} earnings analyst forecast revenue estimate",
    "Financials": "{title} analyst price target earnings forecast",
    "Tech & Science": "{title} release announcement study result data",
}


def _client() -> AsyncTavilyClient:
    return AsyncTavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))


def _format(results: list[dict]) -> str:
    if not results:
        return "No results found."
    parts = []
    for r in results:
        date = r.get("published_date", "")
        title = r.get("title", "")
        content = r.get("content", "")
        parts.append(f"[{date}] {title}: {content}")
    return "\n".join(parts)


async def _search(query: str, days: int | None = None, max_results: int = 5) -> list[dict]:
    try:
        kwargs: dict = {"query": query, "max_results": max_results, "search_depth": "basic"}
        if days is not None:
            kwargs["days"] = days
        resp = await _client().search(**kwargs)
        return resp.get("results", [])
    except Exception:
        return []


async def search_recent_news(title: str) -> str:
    results = await _search(f"{title} news latest update", days=2, max_results=5)
    return _format(results)


async def search_stats_history(title: str) -> str:
    results = await _search(f"{title} odds statistics history base rate", max_results=5)
    return _format(results)


async def search_category(title: str, category: str) -> str:
    template = _CATEGORY_QUERIES.get(category)
    if template:
        query = template.replace("{title}", title)
        results = await _search(query, max_results=5)
        return _format(results)

    # Unknown category: two parallel searches for broader coverage
    q1 = f"{title} {category} prediction expert analysis"
    q2 = f"{title} probability forecast outcome odds"
    r1, r2 = await asyncio.gather(_search(q1, max_results=3), _search(q2, max_results=3))
    return _format(r1 + r2)


async def search_numeric(title: str) -> str:
    results = await _search(f"{title} result official count number", max_results=5)
    if not results:
        results = await _search(f"{title} final tally confirmed", max_results=5)
    return _format(results)
