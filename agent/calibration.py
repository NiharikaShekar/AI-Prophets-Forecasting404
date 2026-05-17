from datetime import datetime, timezone


def _hours_to_close(close_time_str: str) -> float:
    try:
        close_time = datetime.fromisoformat(close_time_str.replace("Z", "+00:00"))
        diff = (close_time - datetime.now(timezone.utc)).total_seconds() / 3600
        return max(0.0, diff)
    except Exception:
        return 48.0


def _shrink_factor(hours: float, evidence_quality: str) -> float:
    if hours > 336:      # > 2 weeks
        base = 0.60
    elif hours > 168:    # 1-2 weeks
        base = 0.40
    elif hours > 48:     # 2-7 days
        base = 0.25
    else:                # < 48 hours — trust our research most
        base = 0.15
    extra = 0.20 if evidence_quality == "Weak" else 0.0
    return min(base + extra, 0.80)


def calibrate_binary(
    p_raw: float,
    prior: float | None,
    close_time: str,
    evidence_quality: str,
) -> float:
    hours = _hours_to_close(close_time)
    shrink = _shrink_factor(hours, evidence_quality)
    anchor = prior if prior is not None else 0.5
    p = (1 - shrink) * p_raw + shrink * anchor
    return max(0.01, min(0.99, p))


def calibrate_multi(
    probs: dict[str, float],
    priors: dict[str, float] | None,
    close_time: str,
    evidence_quality: str,
    outcomes: list[str],
    multi_label: bool = False,
) -> dict[str, float]:
    hours = _hours_to_close(close_time)
    shrink = _shrink_factor(hours, evidence_quality)
    n = len(outcomes)
    default_anchor = 0.5 if multi_label else 1 / n

    calibrated: dict[str, float] = {}
    for o in outcomes:
        p_raw = probs.get(o, default_anchor)
        anchor = (priors or {}).get(o, default_anchor)
        calibrated[o] = max(0.01, min(0.99, (1 - shrink) * p_raw + shrink * anchor))

    if multi_label:
        return calibrated

    total = sum(calibrated.values())
    return {k: v / total for k, v in calibrated.items()}
