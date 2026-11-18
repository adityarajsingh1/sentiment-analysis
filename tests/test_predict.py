import joblib
import pytest

import predict
from sentiment.text import normalize


def test_normalize_matches_corpus_style():
    assert normalize("Great film! Didn’t love the ending, though.") == \
        "great film ! didn't love the ending , though ."
    assert normalize("A bird's-eye-view (sort of)") == "a bird's-eye-view ( sort of )"


@pytest.fixture(scope="module")
def trained():
    return predict.train()


def test_train_reports_the_notebook_score(trained):
    _, acc = trained
    # same split and settings as the notebook's tuned model
    assert acc == pytest.approx(0.868)


def test_predict_obvious_reviews(trained):
    model, _ = trained
    good, bad = predict.predict(model, [
        "A wonderful, moving film with great performances. One of the best this year.",
        "Boring, stupid and a complete waste of time. The worst script I have seen.",
    ])
    assert good.label == "pos" and good.p_pos > 0.6
    assert bad.label == "neg" and bad.p_pos < 0.4


def test_cli_train_then_predict(tmp_path, capsys, trained, monkeypatch):
    monkeypatch.setattr(predict, "train", lambda use_all=False: trained)
    path = tmp_path / "m.joblib"
    assert predict.main(["--model", str(path), "train"]) == 0
    assert isinstance(joblib.load(path), type(trained[0]))
    assert predict.main(["--model", str(path), "the", "worst", "movie", "ever"]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("neg")


def test_cli_without_model(tmp_path):
    assert predict.main(["--model", str(tmp_path / "missing.joblib"), "hello"]) == 1
