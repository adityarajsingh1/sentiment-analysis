"""
Train, save and use the best model from the notebook (tuned TF-IDF + logistic regression).

    python predict.py train                 # fit on the train split, report test accuracy, save
    python predict.py train --all           # fit on all 2,000 reviews instead (no test score)
    python predict.py "what a waste of two hours"
    echo "one of the best films this year" | python predict.py -

Predictions between 0.4 and 0.6 are reported as "unsure": in the notebook's error analysis the
model was right only about 61% of the time in that band, against 99% outside 0.2-0.8.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from sentiment.data import load_reviews, split_reviews
from sentiment.models import tfidf_logreg
from sentiment.text import normalize

DEFAULT_MODEL = Path("models/sentiment.joblib")
UNSURE = (0.4, 0.6)


def best_model() -> Pipeline:
    # the settings the grid search in section 5 of the notebook picked
    return tfidf_logreg(ngram_range=(1, 1), min_df=5, C=10.0)


@dataclass
class Prediction:
    label: str  # "pos", "neg" or "unsure"
    p_pos: float


def train(use_all: bool = False) -> tuple[Pipeline, float | None]:
    """Fit the best model. Returns it with its test accuracy (None when trained on everything)."""
    df = load_reviews()
    if use_all:
        return best_model().fit(df.text, df.label), None
    train_df, test_df = split_reviews(df)
    model = best_model().fit(train_df.text, train_df.label)
    return model, accuracy_score(test_df.label, model.predict(test_df.text))


def predict(model: Pipeline, texts: list[str]) -> list[Prediction]:
    pos = list(model.classes_).index("pos")
    probs = model.predict_proba([normalize(t) for t in texts])[:, pos]
    out = []
    for p in probs:
        label = "unsure" if UNSURE[0] <= p <= UNSURE[1] else ("pos" if p > 0.5 else "neg")
        out.append(Prediction(label, float(p)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="where the model is saved")
    parser.add_argument("--all", action="store_true", help="with 'train': fit on all 2,000 reviews")
    parser.add_argument("text", nargs="+", help="'train', review text, or '-' to read one review from stdin")
    args = parser.parse_args(argv)

    if args.text == ["train"]:
        model, acc = train(use_all=args.all)
        args.model.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, args.model)
        note = f"test accuracy {acc:.3f}" if acc is not None else "trained on all reviews"
        print(f"saved {args.model} ({note})")
        return 0

    if not args.model.exists():
        print(f"no model at {args.model}, run `python predict.py train` first", file=sys.stderr)
        return 1
    texts = [sys.stdin.read()] if args.text == ["-"] else [" ".join(args.text)]
    for p in predict(joblib.load(args.model), texts):
        print(f"{p.label}\tP(pos)={p.p_pos:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
