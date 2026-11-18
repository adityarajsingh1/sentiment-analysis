"""
Turning new text into the same shape as the corpus.

movie_reviews is lowercased, with punctuation split into its own tokens but apostrophes and
hyphens kept inside words ("didn't", "film's", "bird's-eye-view"). A model trained on it should
see new reviews written the same way, or "Great!" and "great !" become different features.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[\w'-]+|[^\w\s]")
_QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


def normalize(text: str) -> str:
    """Lowercase, straighten curly quotes, and space out punctuation like the corpus does."""
    return " ".join(_TOKEN.findall(text.translate(_QUOTES).lower()))
