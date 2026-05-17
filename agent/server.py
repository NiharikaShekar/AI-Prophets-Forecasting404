import logging
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from .event_classifier import classify
from .fallback import market_fallback
from .pipeline import predict as run_pipeline
from .response_utils import truncate_rationale

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="AI-Prophets Forecasting Agent", version="1.0.0")


class EventRequest(BaseModel):
    event_ticker: str = ""
    market_ticker: str = ""
    title: str
    subtitle: str | None = None
    description: str | None = None
    category: str = ""
    rules: str | None = None
    close_time: str = ""
    outcomes: list[str] = []
    resolved_outcome: Any = None


@app.get("/health")
async def health():
    return {"status": "ok"}


_INTERNAL_FIELDS = {"model_reasoning", "judge_critique", "judge_confidence", "evidence_quality"}


def _finalize_response(event: dict, result: dict) -> dict:
    result["rationale"] = truncate_rationale(result.get("rationale", ""))
    classified = classify(event)
    outcomes = event.get("outcomes") or []

    # Convert binary p_yes → probabilities array (required by Prophet Arena API)
    if "p_yes" in result and outcomes:
        p_yes = float(result.pop("p_yes"))
        p_no = round(1.0 - p_yes, 4)
        result["probabilities"] = [
            {"market": outcomes[0], "probability": round(p_yes, 4)},
            {"market": outcomes[1], "probability": p_no},
        ] if len(outcomes) >= 2 else [
            {"market": outcomes[0], "probability": round(p_yes, 4)},
        ]

    if "probabilities" in result:
        total = sum(p["probability"] for p in result["probabilities"])
        logger.info(
            "multi_label=%s sum_probs=%.3f ticker=%s",
            classified.is_multi_label,
            total,
            event.get("market_ticker", ""),
        )

    for field in _INTERNAL_FIELDS:
        result.pop(field, None)

    return result


@app.post("/predict")
async def predict(event: EventRequest):
    logger.info("Predicting: %s — %s", event.market_ticker, event.title)
    payload = event.model_dump()
    try:
        result = await run_pipeline(payload)
        logger.info("Done: %s → %s", event.market_ticker, result)
        return _finalize_response(payload, result)
    except Exception as exc:
        logger.error("Error on %s: %s", event.market_ticker, exc)
        result = await market_fallback(payload)
        return _finalize_response(payload, result)


def main():
    uvicorn.run("agent.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
