from sentiment.data import LABELS, load_reviews


def test_corpus_loads_balanced():
    df = load_reviews()
    assert len(df) == 2000
    assert df["label"].value_counts().to_dict() == {"neg": 1000, "pos": 1000}
    assert set(df["label"]) == set(LABELS)
    assert df["fileid"].is_unique
    assert (df["text"].str.len() > 0).all()
