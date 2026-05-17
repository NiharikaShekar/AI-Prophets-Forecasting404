"""
Sync wrapper around our async pipeline for use with:
    prophet forecast predict --events events.json --local local_agent

The CLI calls predict(event) synchronously, so we run the async pipeline
inside asyncio.run().
"""
import asyncio
from dotenv import load_dotenv

load_dotenv()

from agent.pipeline import predict as _async_predict
from agent.event_classifier import classify


def predict(event: dict) -> dict:
    result = asyncio.run(_async_predict(event))

    outcomes = event.get("outcomes") or []

    # Convert binary p_yes → probabilities array (Prophet Arena format)
    if "p_yes" in result and outcomes:
        p_yes = round(float(result.pop("p_yes")), 4)
        if len(outcomes) >= 2:
            probs = [
                {"market": outcomes[0], "probability": p_yes},
                {"market": outcomes[1], "probability": round(1.0 - p_yes, 4)},
            ]
        else:
            probs = [{"market": outcomes[0], "probability": p_yes}]
        result["probabilities"] = probs

    # Strip internal fields — only return what the API needs
    return {"probabilities": result.get("probabilities", [])}
