"""
The trained models compared in the notebook, as scikit-learn pipelines.

The corpus text is already lowercased and tokenized with spaces between tokens, so the
vectorizers just split on whitespace instead of applying their own token regex. That keeps
punctuation tokens like "!" and "?" and contractions split the way the corpus has them ("don ' t").
"""

from __future__ import annotations

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def _split(text: str) -> list[str]:
    return text.split()


def bow_naive_bayes(ngram_range: tuple[int, int] = (1, 1), min_df: int = 2, alpha: float = 1.0) -> Pipeline:
    """Word counts into multinomial Naive Bayes, the classic text baseline."""
    return Pipeline([
        ("vec", CountVectorizer(tokenizer=_split, token_pattern=None, lowercase=False,
                                ngram_range=ngram_range, min_df=min_df)),
        ("clf", MultinomialNB(alpha=alpha)),
    ])


def tfidf_logreg(ngram_range: tuple[int, int] = (1, 1), min_df: int = 2, C: float = 1.0) -> Pipeline:
    """TF-IDF weighted features (with sublinear tf) into L2-regularised logistic regression."""
    return Pipeline([
        ("vec", TfidfVectorizer(tokenizer=_split, token_pattern=None, lowercase=False,
                                ngram_range=ngram_range, min_df=min_df, sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])
