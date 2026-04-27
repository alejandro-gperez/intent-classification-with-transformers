def clean_text(text: str) -> str:
    """
    Minimal cleaning for transformer input.
    Do NOT remove stopwords or do heavy processing.
    """
    return text.strip().lower()