import pytest

from sentiment.models import bow_naive_bayes, tfidf_logreg

TEXTS = [
    "a great film , truly great !",
    "great acting and a wonderful story",
    "wonderful , i loved it",
    "a bad film , truly bad .",
    "boring story and bad acting",
    "awful , i hated it",
]
LABELS = ["pos", "pos", "pos", "neg", "neg", "neg"]


@pytest.mark.parametrize("make", [bow_naive_bayes, tfidf_logreg])
def test_pipeline_learns_obvious_words(make):
    model = make(min_df=1).fit(TEXTS, LABELS)
    assert list(model.predict(["great and wonderful", "bad and boring"])) == ["pos", "neg"]


def test_vectorizer_keeps_corpus_tokens():
    model = bow_naive_bayes(min_df=1).fit(TEXTS, LABELS)
    vocab = model.named_steps["vec"].vocabulary_
    # punctuation and one-letter words survive because we split on whitespace
    assert "!" in vocab and "i" in vocab and "a" in vocab


def test_bigrams():
    model = tfidf_logreg(ngram_range=(1, 2), min_df=1).fit(TEXTS, LABELS)
    assert "truly bad" in model.named_steps["vec"].vocabulary_
