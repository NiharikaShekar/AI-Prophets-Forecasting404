def truncate_rationale(text: str, max_len: int = 1000) -> str:
    if not text or not text.strip():
        text = "No rationale provided."
    if len(text) <= max_len:
        return text
    suffix = "… [truncated]"
    return text[: max_len - len(suffix)] + suffix
