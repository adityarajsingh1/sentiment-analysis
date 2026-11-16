import pandas as pd

from sentiment.data import load_reviews, split_reviews


def test_split_is_stratified_and_disjoint():
    train, test = split_reviews(load_reviews())
    assert len(train) == 1500 and len(test) == 500
    assert test["label"].value_counts().to_dict() == {"neg": 250, "pos": 250}
    assert not set(train["fileid"]) & set(test["fileid"])


def test_split_is_repeatable():
    df = load_reviews()
    _, a = split_reviews(df)
    _, b = split_reviews(df)
    pd.testing.assert_frame_equal(a, b)
