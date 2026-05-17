"""SEC EDGAR full-text search API — recent filings for Companies category events."""
import logging
from datetime import date, timedelta

import httpx

logger = logging.getLogger(__name__)

_EFTS_URL = "https://efts.sec.gov/LATEST/search-index"
_USER_AGENT = "ForecastingAgent research@forecasting.ai"
_TIMEOUT = httpx.Timeout(connect=3.0, read=8.0, write=3.0, pool=2.0)

_STOP_LOWER = {"will", "the", "a", "an", "in", "of", "to", "by", "for", "be", "vs",
               "and", "its", "their", "has", "have", "report", "quarter", "q1", "q2",
               "q3", "q4", "fiscal", "year", "annual", "revenue", "earnings", "stock",
               "price", "beat", "miss", "exceed", "raise", "cut", "above", "below"}


def _extract_company(title: str) -> str:
    words = title.split()
    caps = [w.rstrip(".,?!") for w in words
            if w and w[0].isupper() and w.lower() not in _STOP_LOWER and len(w) > 1]
    if caps:
        return " ".join(caps[:3])
    fallback = [w.rstrip(".,?!") for w in words if w.lower() not in _STOP_LOWER]
    return " ".join(fallback[:2])


async def get_edgar_filings(title: str) -> str:
    if not title.strip():
        return ""
    company = _extract_company(title)
    if not company:
        return ""

    end_date = date.today()
    start_date = end_date - timedelta(days=90)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                _EFTS_URL,
                headers={"User-Agent": _USER_AGENT},
                params={
                    "q": f'"{company}"',
                    "forms": "8-K,10-K,10-Q",
                    "dateRange": "custom",
                    "startdt": start_date.isoformat(),
                    "enddt": end_date.isoformat(),
                },
            )
            if r.status_code == 403:
                logger.warning("EDGAR: access forbidden — check User-Agent header")
                return ""
            if r.status_code == 429:
                logger.warning("EDGAR: rate-limited")
                return ""
            if r.status_code != 200:
                logger.warning("EDGAR returned HTTP %d for company '%s'", r.status_code, company)
                return ""

            hits = r.json().get("hits", {}).get("hits", [])
            if not hits:
                logger.debug("EDGAR: no filings found for '%s'", company)
                return ""

            lines = [f"SEC EDGAR recent filings for '{company}':"]
            for h in hits[:5]:
                src = h.get("_source", {})
                entity = (src.get("display_names") or ["?"])[0].strip()
                form = src.get("form") or (src.get("root_forms") or ["?"])[0]
                filed = src.get("file_date", "?")
                period = src.get("period_ending", "")
                period_str = f"  (period: {period})" if period else ""
                lines.append(f"  {filed} | {entity} | {form}{period_str}")
            return "\n".join(lines)

    except httpx.TimeoutException:
        logger.warning("EDGAR: timeout for company '%s'", company)
        return ""
    except httpx.ConnectError:
        logger.warning("EDGAR: cannot connect")
        return ""
    except Exception as exc:
        logger.warning("EDGAR unexpected error for '%s': %s", company, exc)
        return ""
