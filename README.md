# sentiment-analysis

Sentiment analysis on movie reviews, done step by step in a notebook: from a first look at the data, through a lexicon baseline and classic machine learning models, to an error analysis of what the best model still gets wrong.

I'm using the NLTK `movie_reviews` corpus: 2,000 IMDb reviews from Pang & Lee (2004), half positive and half negative. It's small enough to train everything on a laptop in seconds, and long-form enough (full reviews, not one-liners) to make the problem interesting.

## Results

All models are scored on the same 500 held-out reviews. Anything tuned (VADER's threshold, the grid searches) was tuned with 5-fold cross-validation on the other 1,500.

| Model | Test accuracy |
|---|---|
| VADER lexicon, threshold 0 | 65.0% |
| VADER lexicon, threshold tuned on train | 69.0% |
| Bag of words + Naive Bayes | 81.2% |
| Bag of words + Naive Bayes, tuned (with bigrams) | 82.4% |
| TF-IDF + logistic regression | 86.0% |
| TF-IDF + logistic regression, tuned | **86.8%** |

What I took away from it:

- **VADER struggles with full-length reviews.** Over ~750 tokens the small positive cues in negative reviews add up, and it calls half of them positive.
- **A trained linear model is the big jump.** Plain word counts with Naive Bayes are 12 points better than the tuned lexicon. TF-IDF with logistic regression adds almost 5 more.
- **Tuning matters less than the choice of model.** It added about a point to each.
- **The remaining errors are about structure, not vocabulary.** Most of the 66 mistakes are low-confidence predictions. Positive reviews that say "not bad" get misread, and for 27 of the errors the last fifth of the review alone gets it right. Section 6 of the notebook goes through them.

## Using the model

[predict.py](predict.py) trains the best model with the settings the grid search picked, saves it, and classifies new text:

```bash
python predict.py train                    # saved models/sentiment.joblib (test accuracy 0.868)
python predict.py "what a waste of two hours"
# neg     P(pos)=0.12
echo "one of the best films this year, a real gem" | python predict.py -
# pos     P(pos)=0.88
```

`python predict.py train --all` fits on all 2,000 reviews instead, for use outside the notebook. Predictions between 0.4 and 0.6 come back as `unsure`, because that's where the model was right only about 61% of the time. Like the model in the notebook, it still misses "not bad": `python predict.py "it was fine i guess, not bad"` gives `neg P(pos)=0.10`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # downloads the corpus on the first run
jupyter lab sentiment_analysis.ipynb
```

CI runs ruff, mypy and the tests, then executes the whole notebook from top to bottom on every push, so the notebook in the repo always runs as committed.

## Contributing

Ideas and pull requests are welcome, especially ones that go after the errors in section 6. [CONTRIBUTING.md](CONTRIBUTING.md) has the setup, the checks to run and a few open ideas.

## License

MIT, see [LICENSE](LICENSE). The movie_reviews corpus is distributed by NLTK and comes from Pang & Lee (2004).
