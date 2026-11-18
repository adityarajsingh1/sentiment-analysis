# Contributing

Thanks for taking a look. This is a small project, so the process is light.

## Getting set up

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt ruff mypy
python -m nltk.downloader movie_reviews stopwords punkt_tab vader_lexicon
```

## Before you open a pull request

Run the same checks CI runs:

```bash
ruff check .
mypy predict.py sentiment
pytest -q
jupyter nbconvert --to notebook --execute --inplace sentiment_analysis.ipynb
```

If you changed the notebook, commit it **with its outputs**. Run it top to bottom first (the last command above does that), so the outputs match the code. Any number you write in a markdown cell or the README should come from an output in the notebook. Please don't round in your favour or quote results from a run you didn't commit.

## Where things live

- `sentiment/data.py`: loading the corpus and the fixed train/test split. Don't change the split's seed or size without a good reason, because every number in the notebook depends on it.
- `sentiment/models.py`: the scikit-learn pipelines. A new model is a new function here that returns a `Pipeline`, plus a test in `tests/test_models.py`.
- `sentiment/text.py`: turning new text into the corpus's tokenized, lowercase format.
- `predict.py`: the command-line tool for the best model.
- `sentiment_analysis.ipynb`: the write-up. New experiments go in as new sections near the end, so the earlier sections still read in order.

## Ideas that would be welcome

- Handling negation, e.g. marking words after "not" until the next punctuation (section 6 shows "not bad" is a real weak spot)
- Weighting the end of a review more heavily (the last 20% alone fixes 27 of the 66 errors)
- A small transformer baseline, as an optional section that CI can skip
- Trying the models on another review corpus, to see how much of the 86.8% is specific to this one

Open an issue first if you're planning something bigger, so we can agree on the shape before you spend time on it.

## Code of conduct

Everyone taking part is expected to follow the [code of conduct](CODE_OF_CONDUCT.md).
