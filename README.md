# RNN Next-Word Prediction - DT086A Machine Learning

Swedish news text dataset → LSTM model that predicts the **6th word** given 5 input words.

## Project Structure

```
rnn_project/
├── step1_data_cleaning.py     ← DONE: Load, clean, save corpus
├── step2_preprocessing.py     ← TODO: Tokenize, build sequences, split
├── step3_model.py             ← TODO: Build & train LSTM
├── step4_evaluate.py          ← TODO: Evaluate & predict
├── cleaned_corpus.txt         ← Output of step 1 (gitignored - large file)
└── README.md
```

## Steps

| # | File | What it does |
|---|------|-------------|
| 1 | `step1_data_cleaning.py` | Strip HTML, remove junk chars, filter short texts |
| 2 | `step2_preprocessing.py` | Tokenize words, build 5→1 sequences, train/val/test split |
| 3 | `step3_model.py` | Embedding + LSTM + Dense model, train and save weights |
| 4 | `step4_evaluate.py` | Accuracy metrics + live next-word prediction demo |

## Requirements

```bash
pip install tensorflow pandas numpy scikit-learn
```

## Dataset
Swedish news articles (25 471 rows). After cleaning: **23 058 articles**, ~4 million tokens, ~228k unique words.

## Run Order

```bash
python step1_data_cleaning.py
python step2_preprocessing.py
python step3_model.py
python step4_evaluate.py
```
