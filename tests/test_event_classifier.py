from agent.event_classifier import classify, is_multi_label


def test_is_multi_label_top5():
    event = {
        "title": "Who finishes in the Eurovision top 5?",
        "rules": "Resolves Yes if act finishes in top 5.",
        "outcomes": ["A", "B", "C", "D", "E"],
    }
    assert is_multi_label(event) is True
    assert classify(event).is_multi_label is True


def test_is_multi_label_winner_override():
    event = {
        "title": "Who wins Ligue 1?",
        "rules": "Winner of the league.",
        "outcomes": ["PSG", "Marseille", "Lyon"],
    }
    assert is_multi_label(event) is False
    assert classify(event).is_multi_label is False


def test_is_multi_label_ambiguous_defaults_false():
    event = {
        "title": "2026 Academy Awards Best Picture",
        "rules": "Resolves to the winning film.",
        "outcomes": ["Film A", "Film B", "Film C"],
    }
    assert is_multi_label(event) is False


def test_numeric_never_multi_label():
    event = {
        "title": "US CPI year-over-year",
        "rules": "Bucket ranges.",
        "outcomes": ["0-2", "2-3", "3-4", "4 or above", "5 or above"],
    }
    classified = classify(event)
    assert classified.type.value == "numeric"
    assert classified.is_multi_label is False
