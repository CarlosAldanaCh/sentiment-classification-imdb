# IMDB Sentiment Analysis — TF-IDF Baselines vs BERT

This project builds and evaluates sentiment classifiers for IMDB movie reviews (binary sentiment: positive/negative).  
We compare classic TF-IDF + linear models against a fine-tuned BERT model, and validate improvements against a Dummy baseline.

## Project goals

- Build a reliable baseline with TF-IDF + Logistic Regression
- Train at least 3 models and compare them using F1-score (main metric)
- Fine-tune BERT for improved performance
- Run a small “live test” with manually written reviews to stress-test edge cases (sarcasm, negation, mixed sentiment)

## Models

- **DummyClassifier** (baseline sanity check)
- **TF-IDF + Logistic Regression**
- **TF-IDF + LinearSVC**
- **BERT (bert-base-uncased)** fine-tuned for sequence classification

## Evaluation

Primary metric: **F1-score**  
We also report: accuracy, precision, recall and confusion matrices.

### Validation results

| Model               | F1     | Precision | Recall | Accuracy |
| ------------------- | ------ | --------- | ------ | -------- |
| BERT                | 0.9206 | 0.9182    | 0.9229 | 0.9205   |
| LinearSVC           | 0.9050 | 0.9010    | 0.9090 | 0.9047   |
| Logistic Regression | 0.8975 | 0.8932    | 0.9018 | 0.8971   |
| DummyClassifier     | 0.0000 | 0.0000    | 0.0000 | 0.5006   |

### Test results

| Model               | F1     | Precision | Recall | Accuracy |
| ------------------- | ------ | --------- | ------ | -------- |
| BERT                | 0.9226 | 0.9158    | 0.9294 | 0.9223   |
| LinearSVC           | 0.8950 | 0.8975    | 0.8925 | 0.8957   |
| Logistic Regression | 0.8905 | 0.8869    | 0.8940 | 0.8904   |
| DummyClassifier     | 0.0000 | 0.0000    | 0.0000 | 0.5019   |

## Live testing (manual reviews)

A small set of manually written reviews was used to test robustness on:

- sarcasm / irony
- negation flips (“I didn’t expect to like it, but…”)
- mixed sentiment (“good acting, messy story”)

> Note: live test results can differ from the formal test set due to small sample size and class imbalance.

## How to run

1. Create and activate a virtual environment
2. Install dependencies
3. Run the notebook end-to-end

## Tech stack

- Python, pandas, numpy
- scikit-learn (TF-IDF, baselines, metrics)
- PyTorch + HuggingFace Transformers (BERT fine-tuning)

## Notes (ES)

- El notebook está documentado mayormente en español.
- El conjunto **test** se mantiene intocable hasta la evaluación final.
- Se incluye baseline con DummyClassifier para validar mejora real.
