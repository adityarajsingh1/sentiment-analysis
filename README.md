# sentiment-analysis

Sentiment analysis on movie reviews, done step by step in a notebook: from a first look at the data, through a lexicon baseline and classic machine learning models, to an error analysis of what the best model still gets wrong.

I'm using the NLTK `movie_reviews` corpus: 2,000 IMDb reviews from Pang & Lee (2004), half positive and half negative. It's small enough to train everything on a laptop in seconds, and long-form enough (full reviews, not one-liners) to make the problem interesting.

## Status

Just started. The notebook ([sentiment_analysis.ipynb](sentiment_analysis.ipynb)) loads the corpus and takes a first look at it; exploring the data comes next.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # downloads the corpus on the first run
```

## License

MIT
