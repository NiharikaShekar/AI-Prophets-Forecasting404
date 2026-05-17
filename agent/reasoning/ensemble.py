import asyncio
import json
import math
import os
import re

from openai import AsyncOpenAI

from ..fallback import binary_from_kalshi_prior, multi_from_kalshi_prior
from .judge import judge_binary, judge_multi

SYSTEM_PROMPT = """You are a world-class superforecaster with deep expertise in calibrated probability estimation.

You MUST follow this exact 7-step reasoning structure:

1. DECOMPOSE: Break the question into 3-5 concrete sub-questions. Answer each one using the provided evidence.
2. BASE RATE: What is the historical base rate for this type of event?
3. ARGUE FOR: The 3 strongest arguments supporting YES / the primary outcome. Weight by recency and source quality.
4. ARGUE AGAINST: The 3 strongest arguments against YES / the primary outcome. Same weighting criteria.
5. SYNTHESIZE: Weigh FOR vs AGAINST. How much does recent evidence move you from the base rate?
6. CALIBRATE: Avoid overconfidence. Only go extreme (>0.85 or <0.15) with very strong, recent, multi-source evidence. When uncertain, stay closer to 0.5 for binary, 1/N for winner-take-all multi-outcome, or moderate per-outcome marginals for multi-label.
7. ESTIMATE: State your final probability.

Then output ONLY valid JSON — no markdown, no extra text."""


def _build_prompt(
    event: dict,
    kalshi_prior: float | dict | None,
    news: str,
    stats: str,
    category_info: str,
    multi_label: bool = False,
) -> str:
    outcomes = event.get("outcomes") or []
    n = len(outcomes)

    if kalshi_prior is None:
        prior_str = "Market price: Not available — use base rates as anchor."
    elif isinstance(kalshi_prior, float):
        prior_str = f"Current Kalshi market price (crowd wisdom): {kalshi_prior:.2%} chance of YES"
    else:
        lines = [f"  {k}: {v:.2%}" for k, v in kalshi_prior.items()]
        prior_str = "Current Kalshi market prices (crowd wisdom):\n" + "\n".join(lines)

    if n <= 2:
        fmt = '{"p_yes": <float 0.01-0.99>, "evidence_quality": "Strong|Moderate|Weak", "rationale": "<your full reasoning>"}'
    else:
        outcome_list = ", ".join(f'"{o}"' for o in outcomes)
        if multi_label:
            fmt = (
                '{"probabilities": [{"market": "<outcome>", "probability": <float>}, ...], '
                '"evidence_quality": "Strong|Moderate|Weak", "rationale": "<your full reasoning>"}\n'
                f"Include ALL outcomes: {outcome_list}\n"
                "Each probability is P(this outcome resolves Yes) as an independent marginal. "
                "Several outcomes may be likely. Do NOT force probabilities to sum to 1.0. "
                "Each must be between 0.01 and 0.99."
            )
        else:
            fmt = (
                '{"probabilities": [{"market": "<outcome>", "probability": <float>}, ...], '
                '"evidence_quality": "Strong|Moderate|Weak", "rationale": "<your full reasoning>"}\n'
                f"Include ALL outcomes: {outcome_list}\n"
                "Probabilities MUST sum to 1.0. Each must be between 0.01 and 0.99."
            )

    return f"""EVENT TO FORECAST
Title: {event.get("title")}
Category: {event.get("category")}
Rules (resolution criterion — do not paraphrase): {event.get("rules") or event.get("description")}
Possible outcomes: {outcomes}
Closes at: {event.get("close_time")}

EVIDENCE GATHERED
{prior_str}

Recent news (last 24-48h — highest priority):
{news}

Historical stats & base rates:
{stats}

Category-specific analysis:
{category_info}

Follow the 7-step reasoning process above. Then output ONLY this JSON (no other text):
{fmt}"""


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return None


def _geo_mean_binary(p_a: float, p_b: float) -> float:
    p_a = max(0.01, min(0.99, p_a))
    p_b = max(0.01, min(0.99, p_b))
    odds_a = p_a / (1 - p_a)
    odds_b = p_b / (1 - p_b)
    combined = math.sqrt(odds_a * odds_b)
    return combined / (1 + combined)


def _geo_mean_multi(
    probs_a: dict,
    probs_b: dict,
    outcomes: list[str],
    multi_label: bool = False,
) -> dict[str, float]:
    result: dict[str, float] = {}
    n = len(outcomes)
    default = 0.5 if multi_label else 1 / n
    for o in outcomes:
        p_a = max(0.001, min(0.999, probs_a.get(o, default)))
        p_b = max(0.001, min(0.999, probs_b.get(o, default)))
        odds_a = p_a / (1 - p_a)
        odds_b = p_b / (1 - p_b)
        combined = math.sqrt(odds_a * odds_b)
        result[o] = max(0.01, min(0.99, combined / (1 + combined)))
    if multi_label:
        return result
    total = sum(result.values())
    return {k: v / total for k, v in result.items()}


def _extremize(p: float, strength: float = 0.05) -> float:
    if p > 0.60:
        return min(0.99, p + strength * (p - 0.5))
    if p < 0.40:
        return max(0.01, p + strength * (p - 0.5))
    return p


async def _call_model(model: str, prompt: str) -> tuple[dict | None, str]:
    """Returns (parsed_json, raw_text). raw_text includes the full model reasoning."""
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    for attempt in range(3):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
                timeout=120,
            )
            text = resp.choices[0].message.content or ""
            # DeepSeek R1 also exposes chain-of-thought in reasoning_content
            reasoning = getattr(resp.choices[0].message, "reasoning_content", None) or ""
            full_text = (f"<thinking>\n{reasoning}\n</thinking>\n\n" + text) if reasoning else text
            result = _extract_json(text)
            if result:
                return result, full_text
        except Exception:
            if attempt < 2:
                await asyncio.sleep(5 * (attempt + 1))
    return None, ""


def _extract_multi_probs(result: dict | None, outcomes: list[str]) -> dict[str, float] | None:
    if not result:
        return None
    raw = result.get("probabilities", [])
    if not isinstance(raw, list):
        return None
    parsed = {}
    for item in raw:
        if "market" in item and "probability" in item:
            market = item["market"]
            if market in outcomes:
                parsed[market] = float(item["probability"])
    return parsed if len(parsed) >= 2 else None


async def run_ensemble(
    event: dict,
    kalshi_prior: float | dict | None,
    news: str,
    stats: str,
    category_info: str,
    event_type: str,
    multi_label: bool = False,
) -> dict:
    model_a = os.environ.get("MODEL_A", "deepseek/deepseek-r1")
    model_b = os.environ.get("MODEL_B", "anthropic/claude-sonnet-4-5")
    outcomes = event.get("outcomes") or []
    n = len(outcomes)

    prompt = _build_prompt(event, kalshi_prior, news, stats, category_info, multi_label)

    # Call both models in parallel
    raw_a, raw_b = await asyncio.gather(
        _call_model(model_a, prompt),
        _call_model(model_b, prompt),
        return_exceptions=True,
    )
    if isinstance(raw_a, Exception):
        raw_a = (None, "")
    if isinstance(raw_b, Exception):
        raw_b = (None, "")

    res_a, text_a = raw_a
    res_b, text_b = raw_b

    evidence_quality = (res_a or res_b or {}).get("evidence_quality", "Moderate")
    rationale = (res_a or res_b or {}).get("rationale", "")

    model_reasoning = {
        model_a: text_a,
        model_b: text_b,
    }

    # Evidence summary for the judge: full evidence the models saw
    if isinstance(kalshi_prior, float):
        prior_line = f"Kalshi market price: {kalshi_prior:.1%} YES"
    elif isinstance(kalshi_prior, dict):
        prior_line = "Kalshi prices: " + ", ".join(f"{k}: {v:.1%}" for k, v in kalshi_prior.items())
    else:
        prior_line = "Kalshi price: unavailable"
    evidence_summary = (
        f"{prior_line}\n\n"
        f"Recent news (Tavily):\n{news[:2000]}\n\n"
        f"Historical stats:\n{stats[:1000]}\n\n"
        f"Category-specific data:\n{category_info[:1000]}"
    )

    # --- Binary ---
    if event_type == "binary":
        p_a = float(res_a["p_yes"]) if res_a and "p_yes" in res_a else None
        p_b = float(res_b["p_yes"]) if res_b and "p_yes" in res_b else None

        judge_critique = ""
        judge_confidence = ""

        if p_a is not None and p_b is not None:
            geo = _geo_mean_binary(p_a, p_b)
            judge_result = await judge_binary(
                event, evidence_summary, p_a, text_a, p_b, text_b
            )
            if judge_result:
                # If both models agree strongly, cap how far the judge can move us
                agreement = abs(p_a - p_b) < 0.20
                if agreement and judge_result["confidence"] == "Low":
                    # Judge is uncertain but models agree — trust the models
                    p_final = _extremize(geo)
                    judge_critique = "[geo-mean used: judge low-confidence override rejected — models agreed] " + judge_result["critique"]
                    judge_confidence = "Moderate"
                else:
                    p_final = judge_result["final_p_yes"]
                    judge_critique = judge_result["critique"]
                    judge_confidence = judge_result["confidence"]
            else:
                # Judge failed — fall back to geometric mean + extremize
                p_final = _extremize(geo)
        elif p_a is not None:
            p_final = p_a
        elif p_b is not None:
            p_final = p_b
        else:
            p_final = binary_from_kalshi_prior(kalshi_prior)

        return {
            "p_yes": p_final,
            "evidence_quality": evidence_quality,
            "rationale": rationale,
            "model_reasoning": model_reasoning,
            "judge_critique": judge_critique,
            "judge_confidence": judge_confidence,
        }

    # --- Multi / Numeric ---
    probs_a = _extract_multi_probs(res_a, outcomes)
    probs_b = _extract_multi_probs(res_b, outcomes)

    judge_critique = ""
    judge_confidence = ""

    if probs_a and probs_b:
        geo_multi = _geo_mean_multi(probs_a, probs_b, outcomes, multi_label)
        # Check if both models agree on the top outcome
        top_a = max(probs_a, key=lambda o: probs_a[o]) if probs_a else None
        top_b = max(probs_b, key=lambda o: probs_b[o]) if probs_b else None
        models_agree_top = top_a == top_b

        judge_result = await judge_multi(
            event, evidence_summary, probs_a, text_a, probs_b, text_b, outcomes, multi_label
        )
        if judge_result:
            if models_agree_top and judge_result["confidence"] == "Low":
                # Both models picked the same winner but judge is uncertain — trust the models
                combined = geo_multi
                judge_critique = "[geo-mean used: judge low-confidence override rejected — models agreed on top outcome] " + judge_result["critique"]
                judge_confidence = "Moderate"
            else:
                combined = judge_result["final_probabilities"]
                judge_critique = judge_result["critique"]
                judge_confidence = judge_result["confidence"]
        else:
            combined = geo_multi
    elif probs_a:
        combined = probs_a
    elif probs_b:
        combined = probs_b
    else:
        combined = multi_from_kalshi_prior(kalshi_prior, outcomes, multi_label)

    default = 0.5 if multi_label else 0.01
    for o in outcomes:
        combined.setdefault(o, default)
    combined = {k: max(0.01, min(0.99, v)) for k, v in combined.items()}
    if not multi_label:
        total = sum(combined.values())
        combined = {k: v / total for k, v in combined.items()}

    return {
        "probabilities": combined,
        "evidence_quality": evidence_quality,
        "rationale": rationale,
        "model_reasoning": model_reasoning,
        "judge_critique": judge_critique,
        "judge_confidence": judge_confidence,
    }
