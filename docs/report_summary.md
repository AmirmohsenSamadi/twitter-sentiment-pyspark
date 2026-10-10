# Report Summary: Sentiment Analysis of Tweets with Logistic Regression and Naive Bayes

## Problem
Tweets are a rich source of public opinion, but they are short and informal, with slang, mentions, links and sarcasm. The goal is to classify tweet sentiment automatically at scale and to compare two classical text classifiers using accuracy, precision, recall, F1 and confusion matrices.

## Data
- Sentiment140: 1,600,000 tweets labelled negative (0) or positive (4).
- Only the label and tweet text are used. Labels are mapped to 0.0 / 1.0.

## Pipeline (Apache Spark on Databricks)
1. **Noise removal:** HTML entities, URLs and `@mentions` removed, and non-alphanumeric characters stripped.
2. **Normalisation:** lowercasing, English stop-word removal (NLTK) and Snowball stemming, implemented as a Spark UDF.
3. **Exploration:** word clouds of the most frequent terms in positive and negative tweets.
4. **Feature engineering:** tokenisation, `CountVectorizer` term counts and IDF weighting (TF-IDF).
5. **Training:** 90/10 random split (fixed seed), then Spark MLlib Logistic Regression and Naive Bayes.
6. **Evaluation:** accuracy, precision, recall and F1 for the positive class, plus confusion matrices on 160,293 test tweets.

## Results
| Metric | Logistic Regression | Naive Bayes |
|---|---:|---:|
| Accuracy | **0.760** | 0.744 |
| Precision | 0.753 | **0.754** |
| Recall | **0.777** | 0.727 |
| F1-score | **0.765** | 0.740 |

## Conclusions
- Logistic Regression is slightly more reliable overall. It makes fewer total errors and finds more positive tweets.
- Naive Bayes is competitive, with nearly identical precision, and remains a fast, simple alternative.
- Both models are limited by bag-of-words features, which ignore word order, negation and sarcasm. Richer features or transformer-based models are the natural next step.
