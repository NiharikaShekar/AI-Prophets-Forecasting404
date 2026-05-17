from agent.calibration import calibrate_multi
from agent.fallback import multi_from_kalshi_prior
from agent.response_utils import truncate_rationale


def test_truncate_rationale_empty():
    result = truncate_rationale("")
    assert result == "No rationale provided."
    assert len(result) >= 1


def test_truncate_rationale_long():
    text = "x" * 1500
    result = truncate_rationale(text)
    assert len(result) <= 1000
    assert result.endswith("… [truncated]")


def test_truncate_rationale_exact_limit():
    text = "a" * 1000
    assert truncate_rationale(text) == text


def test_calibrate_multi_label_no_renormalize():
    outcomes = ["A", "B", "C"]
    probs = {"A": 0.6, "B": 0.5, "C": 0.4}
    result = calibrate_multi(
        probs,
        None,
        "2099-01-01T00:00:00Z",
        "Moderate",
        outcomes,
        multi_label=True,
    )
    assert abs(sum(result.values()) - 1.0) > 0.05


def test_multi_from_kalshi_prior_preserves_sum():
    outcomes = ["A", "B", "C"]
    prior = {"A": 0.3, "B": 0.25, "C": 0.2}
    result = multi_from_kalshi_prior(prior, outcomes, multi_label=True)
    assert abs(sum(result.values()) - 0.75) < 0.01
