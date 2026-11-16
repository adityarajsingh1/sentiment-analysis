# sentiment-analysis

Sentiment analysis on movie reviews, done step by step in a notebook: from a first look at the data, through a lexicon baseline and classic machine learning models, to an error analysis of what the best model still gets wrong.

I'm using the NLTK `movie_reviews` corpus: 2,000 IMDb reviews from Pang & Lee (2004), half positive and half negative. It's small enough to train everything on a laptop in seconds, and long-form enough (full reviews, not one-liners) to make the problem interesting.

## Status

The notebook ([sentiment_analysis.ipynb](sentiment_analysis.ipynb)) loads the corpus, explores it, and compares the first models on a fixed, held-out set of 500 reviews (the trained models are still untuned, so their numbers will move):

| Model | Test accuracy |
|---|---|
| VADER lexicon, threshold 0 | 65.0% |
| VADER lexicon, threshold tuned on train | 69.0% |
| Bag of words + Naive Bayes | 81.2% |
| TF-IDF + logistic regression | 86.0% |

Tuning, a confusion matrix and error analysis come next.

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
