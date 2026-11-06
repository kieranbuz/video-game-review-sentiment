# Video-game review sentiment analysis

Classifying the sentiment of Amazon video-game reviews with scikit-learn, comparing bag-of-words features against sentiment-lexicon scores.

## The task

This is BSc coursework. The task is to classify the sentiment of Amazon video-game reviews.

## Method

Six scikit-learn classifiers were tested: decision tree, k-NN, naive Bayes, random forest, SVM and an MLP neural network. Each was tested in two experiments:

1. **Feature set:** a bag-of-words feature set.
2. **Lexicon:** scores from a 4,980-entry sentiment lexicon. For each review, punctuation is removed, the review is split into words, and the lexicon scores of matching words are each multiplied by 100 and summed. That single value is the classifier's input.

Results are reported as weighted precision, recall and F-measure, plus time taken. The timer covers both training and prediction on the test set.

The scripts in this repo each hold one classifier configuration: `sentiment_analysis.py` uses `CategoricalNB`, and `lexicon_based_sentiment_analysis.py` uses `MLPClassifier(random_state=1000, max_iter=10)`.

These are the final submitted versions of the scripts. During the experiments the classifier import was changed for each algorithm, so only these two configurations remain in the code. The results below come from the coursework write-up; the code has not been re-run for this repo.

## Results

### Feature-set results

| Classifier | F-measure | Time (seconds) |
| --- | --- | --- |
| Decision tree | 0.6761 | 4.7073 |
| k-NN | 0.6450 | 141.6975 |
| Naive Bayes | 0.8229 | 1.5186 |
| Random forest | 0.8167 | 81.8238 |
| SVM | 0.8167 | 131.1304 |
| Neural network (MLP) | 0.8476 | 5.3473 |

### Lexicon results

| Classifier | F-measure | Time (seconds) |
| --- | --- | --- |
| Decision tree | 0.5838 | 0.018 |
| k-NN | 0.6605 | 1.1097 |
| Naive Bayes | 0.6032 | 0.0048 |
| Random forest | 0.5862 | 2.4348 |
| SVM | 0.6633 | 1.1203 |
| Neural network (MLP) | 0.6610 | 0.3317 |

## Limitation

The lexicon scoring method underperformed and needs better pre-processing.

## Data

The review data and the lexicon were supplied by the university module and are not included in this repo.

## How to run

**Not verified.** These steps come from reading the code; they have not been run.

The code needs Python with `pandas`, `numpy` and `scikit-learn`. No versions are pinned.

Both scripts run from top to bottom, with no `main` function or command-line arguments, and read their data files by relative path. Without the data files, they will not run. Put the files in the working directory:

| Script | Data files it reads |
| --- | --- |
| `sentiment_analysis.py` | `reviews_Video_Games_training.csv`, `reviews_Video_Games_test.csv` |
| `lexicon_based_sentiment_analysis.py` | `reviews_Video_Games_training.raw.tsv`, `reviews_Video_Games_test.raw.tsv`, `Games_senti_lexicon.tsv` |

```
python sentiment_analysis.py
python lexicon_based_sentiment_analysis.py
```

Each script prints a `classification_report` and the time taken.

`CW2 - Advanced Task (Filtering Solution).ipynb` removes features that are not in the lexicon from the training and test data. It reads the data from absolute paths, which would need changing before it could run.
