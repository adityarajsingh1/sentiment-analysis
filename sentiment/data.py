"""
Loading the NLTK movie_reviews corpus: 2,000 movie reviews from IMDb
(Pang & Lee, 2004), 1,000 labelled positive and 1,000 negative.
"""

from __future__ import annotations

import nltk
import pandas as pd

LABELS = ("neg", "pos")


def ensure_corpus(name: str = "movie_reviews") -> None:
    """Download an NLTK corpus the first time it's needed."""
    try:
        nltk.data.find(f"corpora/{name}")
    except LookupError:
        nltk.download(name, quiet=True)


def load_reviews() -> pd.DataFrame:
    """One row per review: file id, label ("pos" / "neg") and the raw text.

    The corpus is already lowercased and tokenized (tokens separated by spaces),
    so `text` is that token stream joined back together.
    """
    ensure_corpus()
    from nltk.corpus import movie_reviews

    rows = [
        {"fileid": fileid, "label": label, "text": movie_reviews.raw(fileid)}
        for label in LABELS
        for fileid in movie_reviews.fileids(label)
    ]
    return pd.DataFrame(rows)
