"""Reddit API — social media sentiment for Mentions/Culture events."""
import logging
import os
from base64 import b64encode

import httpx

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_SEARCH_URL = "https://oauth.reddit.com/search.json"
_USER_AGENT = "ForecastingAgent/1.0 (by /u/forecasting_bot)"
_TIMEOUT = httpx.Timeout(connect=3.0, read=7.0, write=3.0, pool=2.0)


async def _get_token(client_id: str, client_secret: str) -> str | None:
    creds = b64encode(f"{client_id}:{client_secret}".encode()).decode()
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.post(
                _TOKEN_URL,
                headers={"Authorization": f"Basic {creds}", "User-Agent": _USER_AGENT},
                data={"grant_type": "client_credentials"},
            )
            if r.status_code == 401:
                logger.warning("Reddit: invalid client credentials")
                return None
            if r.status_code != 200:
                logger.warning("Reddit token endpoint returned HTTP %d", r.status_code)
                return None
            token = r.json().get("access_token")
            if not token:
                logger.warning("Reddit: no access_token in response")
            return token
    except httpx.TimeoutException:
        logger.warning("Reddit: timeout fetching token")
        return None
    except httpx.ConnectError:
        logger.warning("Reddit: cannot connect to token endpoint")
        return None
    except Exception as exc:
        logger.warning("Reddit token unexpected error: %s", exc)
        return None


async def get_reddit_mentions(title: str) -> str:
    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return ""
    if not title.strip():
        return ""

    token = await _get_token(client_id, client_secret)
    if not token:
        return ""

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                _SEARCH_URL,
                headers={"Authorization": f"bearer {token}", "User-Agent": _USER_AGENT},
                params={"q": title[:80], "sort": "new", "t": "week",
                        "limit": 10, "type": "link"},
            )
            if r.status_code == 429:
                logger.warning("Reddit: rate-limited")
                return ""
            if r.status_code == 403:
                logger.warning("Reddit: access forbidden — token may have expired")
                return ""
            if r.status_code != 200:
                logger.warning("Reddit search returned HTTP %d", r.status_code)
                return ""
            posts = r.json().get("data", {}).get("children", [])
            if not posts:
                logger.debug("Reddit: no posts found for '%s'", title[:40])
                return ""

            total_score = sum(p["data"].get("score", 0) for p in posts)
            total_comments = sum(p["data"].get("num_comments", 0) for p in posts)
            lines = [
                f"Reddit mentions (last 7 days, {len(posts)} posts):",
                f"  Total upvotes: {total_score:,}  |  Total comments: {total_comments:,}",
            ]
            for p in posts[:4]:
                d = p["data"]
                sub = d.get("subreddit_name_prefixed", "r/?")
                post_title = d.get("title", "")[:70]
                score = d.get("score", 0)
                lines.append(f"  [{sub}] {post_title} (↑{score})")
            return "\n".join(lines)

    except httpx.TimeoutException:
        logger.warning("Reddit: timeout during search")
        return ""
    except httpx.ConnectError:
        logger.warning("Reddit: cannot connect to search endpoint")
        return ""
    except Exception as exc:
        logger.warning("Reddit search unexpected error: %s", exc)
        return ""
