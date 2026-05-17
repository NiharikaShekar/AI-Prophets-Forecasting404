"""
Judge agent: reviews both model predictions and produces a final calibrated probability.

Called after Model A and Model B complete. Sees their full reasoning and arbitrates:
- Catches overconfidence, anchoring bias, ignored evidence, or correlated errors
- Produces a final probability that replaces the geometric mean
- Writes a ≤1000-char critique explaining the decision
Falls back gracefully if the judge call fails.
"""
import asyncio
import json
import logging
import os
import re

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = """You are a senior forecasting auditor. Two AI models have made probability predictions for the same event. Your job is to produce the final calibrated probability.

## AGREEMENT RULE (most important)
If both models point in the same direction AND their probabilities are within 0.20 of each other, ACCEPT their consensus. Do NOT flatten or override it without a specific, named piece of evidence that contradicts them. Agreement between two independent models is strong signal, not a reason for suspicion.

## WHEN TO OVERRIDE
Only override the consensus when you can point to a specific flaw:
- A factual error: the models cited something wrong or non-existent
- A missed resolution criterion: the market resolves differently from what the models assumed
- Evidence in the KEY EVIDENCE block that directly contradicts both conclusions
Do NOT override just because the prediction feels extreme. Correct predictions about resolved events ARE extreme.

## CALIBRATION RULES
- "High" confidence: both models agree strongly (gap < 0.15) AND evidence is consistent
- "Moderate" confidence: models agree but gap is 0.15–0.30, OR evidence is mixed
- "Low" confidence: models disagree significantly (gap > 0.30) OR evidence directly contradicts both
- Never assign Low confidence just because outcomes are uncertain in general

## FOR MULTI-OUTCOME EVENTS
The model with the higher-probability top pick is usually right when both models agree on the winner. Preserve the top pick's relative advantage — do not flatten the distribution across all outcomes.

Your final probability should be the most defensible number given the evidence.
Output ONLY valid JSON — no markdown, no text outside the JSON."""


def _excerpt(text: str, max_chars: int = 2000) -> str:
    if not text:
        return "(no reasoning)"
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"… [{len(text)} chars total]"


def _build_binary_prompt(
    event: dict,
    evidence_summary: str,
    p_a: float,
    reasoning_a: str,
    p_b: float,
    reasoning_b: str,
) -> str:
    rules = event.get("rules") or event.get("description") or ""
    return f"""EVENT
Title: {event.get("title")}
Category: {event.get("category")}
Resolution: {rules[:400]}
Closes: {event.get("close_time")}

━━━ FULL EVIDENCE (same as what the models saw) ━━━
{evidence_summary[:4000]}

━━━ MODEL A (DeepSeek R1) ━━━
Prediction: p_yes = {p_a:.4f}  ({p_a:.1%})
Full reasoning:
{_excerpt(reasoning_a)}

━━━ MODEL B (Claude Sonnet) ━━━
Prediction: p_yes = {p_b:.4f}  ({p_b:.1%})
Full reasoning:
{_excerpt(reasoning_b)}

━━━ YOUR TASK ━━━
You have the same evidence the models had. Use it to verify their conclusions.
If A and B agree (same direction, gap < 0.20): validate their consensus and return a number close to their average. Only override if you can name a specific factual flaw found in the evidence above.
If A and B disagree: weigh the evidence directly and pick the better-supported estimate.

Output this JSON exactly:
{{"final_p_yes": <float 0.01-0.99>, "confidence": "High|Moderate|Low", "critique": "<≤1000 chars: did you agree or override? cite the specific evidence that drove your decision>"}}"""


def _build_multi_prompt(
    event: dict,
    evidence_summary: str,
    probs_a: dict,
    reasoning_a: str,
    probs_b: dict,
    reasoning_b: str,
    outcomes: list[str],
    multi_label: bool,
) -> str:
    rules = event.get("rules") or event.get("description") or ""
    outcome_list = ", ".join(f'"{o}"' for o in outcomes)
    norm_note = (
        "Each probability is an INDEPENDENT marginal P(this outcome resolves Yes). Do NOT force them to sum to 1.0."
        if multi_label
        else "Probabilities MUST sum to 1.0."
    )
    fmt = (
        f'{{"final_probabilities": [{{"market": "<outcome>", "probability": <float>}}, ...], '
        f'"confidence": "High|Moderate|Low", '
        f'"critique": "<≤1000 chars>"}}\n'
        f"Include ALL outcomes: {outcome_list}\n{norm_note}"
    )
    return f"""EVENT
Title: {event.get("title")}
Category: {event.get("category")}
Resolution: {rules[:400]}
Outcomes: {outcomes}
Closes: {event.get("close_time")}

━━━ FULL EVIDENCE (same as what the models saw) ━━━
{evidence_summary[:4000]}

━━━ MODEL A (DeepSeek R1) ━━━
Prediction: {json.dumps(probs_a, indent=2)}
Full reasoning:
{_excerpt(reasoning_a)}

━━━ MODEL B (Claude Sonnet) ━━━
Prediction: {json.dumps(probs_b, indent=2)}
Full reasoning:
{_excerpt(reasoning_b)}

━━━ YOUR TASK ━━━
You have the same evidence the models had. Use it to verify their conclusions.
If A and B agree on the top outcome: validate their consensus. Preserve the top pick's probability advantage — do NOT flatten the distribution. Only override if you can name a specific factual flaw found in the evidence above.
If A and B disagree: weigh the evidence directly and pick the better-supported estimate.

Output this JSON exactly:
{fmt}"""


def _extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


async def _call_judge(prompt: str) -> dict | None:
    model = os.environ.get("MODEL_JUDGE", "anthropic/claude-sonnet-4-5")
    try:
        api_key = os.environ["OPENROUTER_API_KEY"]
    except KeyError:
        logger.error("Judge: OPENROUTER_API_KEY not set")
        return None

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    for attempt in range(2):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1200,
                timeout=120,
            )
            text = resp.choices[0].message.content or ""
            result = _extract_json(text)
            if result:
                logger.info("Judge (%s) completed on attempt %d", model, attempt + 1)
                return result
            logger.warning("Judge: could not parse JSON from response (attempt %d)", attempt + 1)
        except Exception as exc:
            if attempt == 0:
                logger.warning("Judge: attempt 1 failed (%s) — retrying", exc)
                await asyncio.sleep(3)
            else:
                logger.warning("Judge: all attempts failed — %s", exc)
    return None


async def judge_binary(
    event: dict,
    evidence_summary: str,
    p_a: float,
    reasoning_a: str,
    p_b: float,
    reasoning_b: str,
) -> dict | None:
    """
    Returns {"final_p_yes": float, "confidence": str, "critique": str}
    or None if the judge call fails (caller should fall back to geo mean).
    """
    prompt = _build_binary_prompt(event, evidence_summary, p_a, reasoning_a, p_b, reasoning_b)
    result = await _call_judge(prompt)
    if result is None:
        return None

    try:
        p = float(result["final_p_yes"])
        p = max(0.01, min(0.99, p))
        critique = str(result.get("critique", ""))[:1000]
        confidence = str(result.get("confidence", "Moderate"))
        return {"final_p_yes": p, "confidence": confidence, "critique": critique}
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Judge binary: unexpected result shape — %s: %s", exc, result)
        return None


async def judge_multi(
    event: dict,
    evidence_summary: str,
    probs_a: dict,
    reasoning_a: str,
    probs_b: dict,
    reasoning_b: str,
    outcomes: list[str],
    multi_label: bool = False,
) -> dict | None:
    """
    Returns {"final_probabilities": dict[str,float], "confidence": str, "critique": str}
    or None if the judge call fails.
    """
    prompt = _build_multi_prompt(
        event, evidence_summary, probs_a, reasoning_a, probs_b, reasoning_b, outcomes, multi_label
    )
    result = await _call_judge(prompt)
    if result is None:
        return None

    try:
        raw = result.get("final_probabilities", [])
        if not isinstance(raw, list):
            raise ValueError("final_probabilities must be a list")
        probs: dict[str, float] = {}
        for item in raw:
            market = item["market"]
            if market in outcomes:
                probs[market] = max(0.01, min(0.99, float(item["probability"])))
        if len(probs) < max(2, len(outcomes) - 1):
            logger.warning("Judge multi: only %d/%d outcomes returned", len(probs), len(outcomes))
            return None
        # Fill any missing outcomes with uniform share
        n = len(outcomes)
        for o in outcomes:
            probs.setdefault(o, 0.5 if multi_label else 1 / n)
        if not multi_label:
            total = sum(probs.values())
            if total > 0:
                probs = {k: v / total for k, v in probs.items()}

        critique = str(result.get("critique", ""))[:1000]
        confidence = str(result.get("confidence", "Moderate"))
        return {"final_probabilities": probs, "confidence": confidence, "critique": critique}
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("Judge multi: unexpected result shape — %s: %s", exc, result)
        return None
