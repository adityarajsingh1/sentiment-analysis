# sentiment-analysis

Sentiment analysis on movie reviews, done step by step in a notebook: from a first look at the data, through a lexicon baseline and classic machine learning models, to an error analysis of what the best model still gets wrong.

I'm using the NLTK `movie_reviews` corpus: 2,000 IMDb reviews from Pang & Lee (2004), half positive and half negative. It's small enough to train everything on a laptop in seconds, and long-form enough (full reviews, not one-liners) to make the problem interesting.

## Status

The notebook ([sentiment_analysis.ipynb](sentiment_analysis.ipynb)) loads the corpus, explores it, and compares models on a fixed, held-out set of 500 reviews. Tuning was done with 5-fold cross-validation on the other 1,500:

| Model | Test accuracy |
|---|---|
| VADER lexicon, threshold 0 | 65.0% |
| VADER lexicon, threshold tuned on train | 69.0% |
| Bag of words + Naive Bayes | 81.2% |
| Bag of words + Naive Bayes, tuned (with bigrams) | 82.4% |
| TF-IDF + logistic regression | 86.0% |
| TF-IDF + logistic regression, tuned | **86.8%** |

The last section looks at the 66 reviews the best model gets wrong. Most are ones it was unsure about. Positive reviews with phrases like "not bad" are a clear weak spot, and for about 40% of the errors the verdict sits at the end of the review.

Still to come: a small `predict.py` to train, save and use the best model, and contributor docs.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # downloads the corpus on the first run
jupyter lab sentiment_analysis.ipynb
```

CI runs the tests and executes the whole notebook from top to bottom on every push, so the notebook in the repo always runs as committed.

## License

MIT
