"""The Odds API — sports betting odds for Sports category events."""
import logging
import os

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://api.the-odds-api.com/v4"
_TIMEOUT = httpx.Timeout(connect=3.0, read=7.0, write=3.0, pool=2.0)

_SPORT_MAP = [
    (["nfl", "super bowl", "patriots", "chiefs", "american football"], "americanfootball_nfl"),
    (["nba", "basketball", "lakers", "celtics", "knicks", "warriors"], "basketball_nba"),
    (["mlb", "baseball", "yankees", "dodgers", "red sox", "mets"], "baseball_mlb"),
    (["nhl", "hockey", "stanley cup", "bruins", "rangers", "penguins"], "icehockey_nhl"),
    (["ufc", "mma", "bellator", "fight", "knockout", "diaz", "mcgregor", "rousey", "perry", "carano"], "mma_mixed_martial_arts"),
    (["premier league", "la liga", "bundesliga", "serie a", "champions league", "soccer"], "soccer_epl"),
    (["tennis", "atp", "wta", "wimbledon", "french open", "us open", "australian open"], "tennis_atp_french_open"),
    (["golf", "pga", "masters", "open championship", "ryder cup"], "golf_pga_championship"),
    (["college football", "ncaaf", "cfp"], "americanfootball_ncaaf"),
    (["college basketball", "ncaa", "march madness"], "basketball_ncaab"),
]


def _detect_sport(title: str) -> str | None:
    t = title.lower()
    for keywords, sport_key in _SPORT_MAP:
        if any(kw in t for kw in keywords):
            return sport_key
    return None


def _format_odds(bookmakers: list[dict]) -> str:
    seen: dict[str, list[float]] = {}
    for bm in bookmakers[:5]:
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes", []):
                name = outcome.get("name", "")
                price = outcome.get("price", 0)
                seen.setdefault(name, []).append(price)
    lines = []
    for name, prices in seen.items():
        avg = sum(prices) / len(prices)
        implied = 100 / (avg + 100) if avg > 0 else abs(avg) / (abs(avg) + 100)
        lines.append(f"  {name}: avg odds {avg:+.0f} → implied {implied:.1%}")
    return "Betting odds (The Odds API):\n" + "\n".join(lines) if lines else ""


async def get_sports_odds(title: str) -> str:
    api_key = os.environ.get("ODDS_API_KEY", "")
    if not api_key:
        return ""
    if not title.strip():
        return ""
    sport = _detect_sport(title)
    if not sport:
        logger.debug("Odds API: no sport detected in '%s'", title[:50])
        return ""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_BASE}/sports/{sport}/odds/",
                params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
            )
            if r.status_code == 401:
                logger.warning("Odds API: invalid API key")
                return ""
            if r.status_code == 422:
                logger.debug("Odds API: sport '%s' not currently available", sport)
                return ""
            if r.status_code == 429:
                logger.warning("Odds API: quota exceeded")
                return ""
            if r.status_code != 200:
                logger.warning("Odds API returned HTTP %d for sport %s", r.status_code, sport)
                return ""
            games = r.json()
            if not isinstance(games, list):
                logger.warning("Odds API: unexpected response format")
                return ""
            title_lower = title.lower()
            for game in games:
                home = game.get("home_team", "").lower()
                away = game.get("away_team", "").lower()
                if home in title_lower or away in title_lower:
                    result = _format_odds(game.get("bookmakers", []))
                    if result:
                        return result
            # No exact match — return first game in the sport as context
            if games:
                return _format_odds(games[0].get("bookmakers", []))
            return ""
    except httpx.TimeoutException:
        logger.warning("Odds API: timeout for sport %s", sport)
        return ""
    except httpx.ConnectError:
        logger.warning("Odds API: cannot connect")
        return ""
    except Exception as exc:
        logger.warning("Odds API unexpected error: %s", exc)
        return ""
