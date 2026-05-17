import logging
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from .pipeline import predict as run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AI-Prophets Forecasting Agent", version="1.0.0")


class EventRequest(BaseModel):
    event_ticker: str
    market_ticker: str
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str
    rules: str | None = None
    close_time: str
    outcomes: list[str] | None = None
    resolved_outcome: Any = None


def _fallback(outcomes: list[str]) -> dict:
    if len(outcomes) <= 2:
        return {"p_yes": 0.5, "rationale": "Internal error — returning neutral fallback."}
    n = len(outcomes)
    return {
        "probabilities": [
            {"market": o, "probability": round(1 / n, 4)} for o in outcomes
        ],
        "rationale": "Internal error — returning uniform fallback.",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(event: EventRequest):
    logger.info("Predicting: %s — %s", event.market_ticker, event.title)
    outcomes = event.outcomes or []
    try:
        result = await run_pipeline(event.model_dump())
        logger.info("Done: %s → %s", event.market_ticker, result)
        return result
    except Exception as exc:
        logger.error("Error on %s: %s", event.market_ticker, exc)
        return _fallback(outcomes)


def main():
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
