"""NOAA Climate Data API — weather data for Climate category events."""
import logging
import os
from datetime import date, timedelta

import httpx

logger = logging.getLogger(__name__)

_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
_TIMEOUT = httpx.Timeout(connect=3.0, read=10.0, write=3.0, pool=2.0)

_CITY_STATIONS = {
    "new york": "GHCND:USW00094728", "nyc": "GHCND:USW00094728",
    "los angeles": "GHCND:USW00023174", "la": "GHCND:USW00023174",
    "chicago": "GHCND:USW00094846",
    "houston": "GHCND:USW00012960",
    "phoenix": "GHCND:USW00023183",
    "philadelphia": "GHCND:USW00013739",
    "dallas": "GHCND:USW00003927",
    "miami": "GHCND:USW00012839",
    "atlanta": "GHCND:USW00013874",
    "seattle": "GHCND:USW00024233",
    "boston": "GHCND:USW00014739",
    "denver": "GHCND:USW00003017",
    "minneapolis": "GHCND:USW00014922",
    "las vegas": "GHCND:USW00023169",
    "washington": "GHCND:USW00013743", "dc": "GHCND:USW00013743",
    "portland": "GHCND:USW00024229",
}
_DEFAULT_STATION = "GHCND:USW00094728"  # NYC fallback


def _detect_station(title: str) -> str:
    t = title.lower()
    for city, station in _CITY_STATIONS.items():
        if city in t:
            return station
    return _DEFAULT_STATION


async def get_climate_data(title: str) -> str:
    token = os.environ.get("NOAA_TOKEN", "")
    if not token:
        return ""
    if not title.strip():
        return ""

    station = _detect_station(title)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(
                f"{_BASE}/data",
                headers={"token": token},
                params={
                    "datasetid": "GHCND",
                    "stationid": station,
                    "datatypeid": "TMAX,TMIN,PRCP",
                    "startdate": start_date.isoformat(),
                    "enddate": end_date.isoformat(),
                    "limit": 30,
                    "units": "standard",
                },
            )
            if r.status_code == 400:
                logger.warning("NOAA: bad request — station %s may be invalid", station)
                return ""
            if r.status_code == 401:
                logger.warning("NOAA: invalid token")
                return ""
            if r.status_code == 429:
                logger.warning("NOAA: rate-limited")
                return ""
            if r.status_code != 200:
                logger.warning("NOAA returned HTTP %d for station %s", r.status_code, station)
                return ""

            results = r.json().get("results", [])
            if not results:
                logger.debug("NOAA: no data for station %s in date range", station)
                return ""

            by_date: dict[str, dict] = {}
            for item in results:
                d = item.get("date", "")[:10]
                by_date.setdefault(d, {})[item.get("datatype", "")] = item.get("value")

            lines = ["NOAA climate data (last 30 days):"]
            for d in sorted(by_date.keys())[-7:]:
                day = by_date[d]
                parts = []
                if day.get("TMAX") is not None:
                    parts.append(f"High: {day['TMAX']/10:.1f}°C")
                if day.get("TMIN") is not None:
                    parts.append(f"Low: {day['TMIN']/10:.1f}°C")
                if day.get("PRCP") is not None:
                    parts.append(f"Precip: {day['PRCP']/10:.1f}mm")
                lines.append(f"  {d}: " + "  |  ".join(parts))
            return "\n".join(lines)

    except httpx.TimeoutException:
        logger.warning("NOAA: timeout for station %s", station)
        return ""
    except httpx.ConnectError:
        logger.warning("NOAA: cannot connect")
        return ""
    except Exception as exc:
        logger.warning("NOAA unexpected error: %s", exc)
        return ""
