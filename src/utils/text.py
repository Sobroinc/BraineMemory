"""Text processing utilities."""

import re
import unicodedata
from functools import lru_cache

import tiktoken


# ─────────────────────────────────────────────────────────────────────────
# Token Counting
# ─────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_encoder() -> tiktoken.Encoding:
    """Get tiktoken encoder (cached)."""
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """
    Count tokens in text using tiktoken.

    Uses cl100k_base encoding (GPT-4, text-embedding-3).
    """
    if not text:
        return 0
    encoder = _get_encoder()
    return len(encoder.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within token limit.

    Returns the original text if it fits, otherwise truncates
    to the maximum number of tokens.
    """
    if not text:
        return ""

    encoder = _get_encoder()
    tokens = encoder.encode(text)

    if len(tokens) <= max_tokens:
        return text

    return encoder.decode(tokens[:max_tokens])


def estimate_tokens(text: str) -> int:
    """
    Fast token estimation (approximate).

    Uses 4 characters per token heuristic.
    Use count_tokens() for accurate counting.
    """
    return len(text) // 4


def normalize_text(text: str) -> str:
    """
    Normalize text for entity matching.

    - Lowercase
    - Remove accents
    - Collapse whitespace
    - Remove punctuation (optional)
    """
    # Lowercase
    text = text.lower()

    # Remove accents (NFD decomposition)
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def detect_language(text: str) -> str:
    """
    Simple language detection based on character ranges.

    Returns: 'ru', 'en', 'fr', or 'multi'
    """
    if not text:
        return "multi"

    # Count character types
    cyrillic = 0
    latin = 0
    french_chars = 0

    for char in text:
        if "\u0400" <= char <= "\u04ff":
            cyrillic += 1
        elif "a" <= char.lower() <= "z":
            latin += 1
        elif char in "àâäéèêëïîôùûüœæç":
            french_chars += 1

    total = cyrillic + latin + french_chars
    if total == 0:
        return "multi"

    # Determine dominant language
    if cyrillic / total > 0.5:
        return "ru"
    elif french_chars / total > 0.1:
        return "fr"
    elif latin / total > 0.5:
        return "en"
    else:
        return "multi"


def extract_entities_simple(text: str) -> list[dict[str, str]]:
    """
    Simple entity extraction using regex patterns.

    This is a fallback when LLM extraction is not available.
    """
    entities = []

    # Email pattern
    emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
    for email in emails:
        entities.append({"name": email, "type": "email"})

    # URL pattern
    urls = re.findall(r"https?://[^\s<>\"{}|\\^`\[\]]+", text)
    for url in urls:
        entities.append({"name": url, "type": "url"})

    # Phone patterns (simple)
    phones = re.findall(r"\+?[0-9]{1,3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{3}[-.\s]?[0-9]{2,4}", text)
    for phone in phones:
        entities.append({"name": phone, "type": "phone"})

    # Date patterns (simple)
    dates = re.findall(r"\d{1,2}[./]\d{1,2}[./]\d{2,4}", text)
    for date in dates:
        entities.append({"name": date, "type": "date"})

    # Dimension patterns (for drawings)
    dims = re.findall(r"[ØR]?\d+(?:[.,]\d+)?(?:\s*[±]\s*\d+(?:[.,]\d+)?)?(?:\s*(?:мм|mm|см|cm|м|m))?", text)
    for dim in dims:
        if len(dim) > 2:  # Skip single digits
            entities.append({"name": dim, "type": "dimension"})

    return entities
