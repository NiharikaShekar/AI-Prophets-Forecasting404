"""TMDB (The Movie Database) API — for Culture category events (movies, TV, awards)."""
import os

import httpx

_BASE = "https://api.themoviedb.org/3"

# Leading question words to strip before extracting entity
_LEAD_STRIP = {"will", "does", "is", "are", "has", "have", "did", "can", "would", "do"}


def _extract_entity(title: str) -> str:
    """
    Pull the entity name (movie/show/person) out of an event title.
    E.g. 'Will Deadpool win best picture?' → 'Deadpool'
         'Will Beyonce win Grammy 2026?' → 'Beyonce'
    Strategy: strip leading question word, then take the first 1-2 words
    that start with an uppercase letter (proper nouns).
    """
    words = title.split()
    if not words:
        return title[:30]

    # Drop leading question word (Will, Does, Is, ...)
    start = 0
    if words[0].lower().rstrip("?.,!") in _LEAD_STRIP:
        start = 1

    # Collect consecutive proper-noun words (start uppercase, not all-caps acronyms > 2 chars)
    proper = []
    for w in words[start:start + 5]:
        clean = w.rstrip("?.,!")
        if clean and clean[0].isupper():
            proper.append(clean)
            if len(proper) == 2:
                break
        else:
            break

    return " ".join(proper) if proper else " ".join(words[start:start + 2])


async def get_tmdb_data(title: str) -> str:
    api_key = os.environ.get("TMDB_API_KEY", "")
    if not api_key:
        return ""

    entity = _extract_entity(title)

    # Short key (32 hex chars) = v3 → api_key query param
    # Long JWT (starts eyJ) = v4 Bearer token → Authorization header
    if api_key.startswith("eyJ"):
        headers = {"Authorization": f"Bearer {api_key}", "accept": "application/json"}
        params_extra: dict = {}
    else:
        headers = {"accept": "application/json"}
        params_extra = {"api_key": api_key}

    try:
        async with httpx.AsyncClient(timeout=7.0) as client:
            # /search/multi searches movies + TV + people in one call
            params = {"query": entity, "page": 1, **params_extra}
            r = await client.get(f"{_BASE}/search/multi", params=params, headers=headers)
            if r.status_code != 200:
                return ""
            results = r.json().get("results", [])
            if not results:
                return ""

        lines = [f"TMDB data for '{entity}':"]
        for item in results[:5]:
            media = item.get("media_type", "")
            if media == "movie":
                t = item.get("title", "")
                year = (item.get("release_date") or "")[:4]
                rating = item.get("vote_average", 0)
                votes = item.get("vote_count", 0)
                pop = item.get("popularity", 0)
                lines.append(f"  [Movie] {t} ({year}): rating {rating:.1f}/10 ({votes:,} votes), popularity {pop:.0f}")
            elif media == "tv":
                name = item.get("name", "")
                rating = item.get("vote_average", 0)
                votes = item.get("vote_count", 0)
                pop = item.get("popularity", 0)
                lines.append(f"  [TV] {name}: rating {rating:.1f}/10 ({votes:,} votes), popularity {pop:.0f}")
            elif media == "person":
                name = item.get("name", "")
                pop = item.get("popularity", 0)
                dept = item.get("known_for_department", "")
                known = [k.get("title") or k.get("name", "") for k in item.get("known_for", [])[:3]]
                lines.append(f"  [Person] {name} ({dept}): popularity {pop:.0f}, known for: {', '.join(known)}")

        return "\n".join(lines)
    except Exception:
        return ""
