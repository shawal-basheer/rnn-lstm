# RNN Next-Word Prediction - DT086A Machine Learning

An LSTM model trained on Swedish news text that predicts the 6th word given the 5 words before it.

## Project structure

```
rnn-lstm/
├── step1_data_cleaning.py     Load, clean and save the corpus
├── step2_preprocessing.py     Tokenize, build sequences, split
├── step3_model.py             Build and train the LSTM
├── step4_evaluate.py          Evaluate and predict
├── demo_app.py                Flask interface for live predictions
├── cleaned_corpus.txt         Output of step 1 (gitignored, large)
└── README.md
```

## Pipeline

| #   | File                     | What it does                                                        |
| --- | ------------------------ | ------------------------------------------------------------------- |
| 1   | `step1_data_cleaning.py` | Strip HTML, remove junk characters, filter out very short texts     |
| 2   | `step2_preprocessing.py` | Tokenize words, build 5-to-1 sequences, train/validation/test split |
| 3   | `step3_model.py`         | Embedding + LSTM + Dense model, train and save weights              |
| 4   | `step4_evaluate.py`      | Accuracy metrics and live next-word prediction                      |

## Dataset

Swedish news articles, 25,471 rows.

After cleaning: 23,058 articles, roughly 4 million tokens and about 228,000 unique words. The vocabulary is capped at the 20,000 most frequent words, with everything else mapped to an out-of-vocabulary token.

## Approach

Each article is turned into overlapping windows of six words. The first five are the input, the sixth is the target.

```
"det är en riktig mastodonträttegång som inleds"

X = [det, är, en, riktig, mastodonträttegång]   y = som
X = [är, en, riktig, mastodonträttegång, som]   y = inleds
```

Targets are stored as plain integer word IDs rather than one-hot vectors. One-hot encoding 4 million sequences against a 20,000 word vocabulary would need roughly 587 GB of memory. Using `sparse_categorical_crossentropy` takes integer labels directly, which is the same computation with none of the cost.

The model is an embedding layer (100 dimensions), a single LSTM layer (64 units), a dense ReLU layer, dropout, and a softmax output over the full vocabulary.

## Results

Trained on a 50,000 sequence subsample of the 3.1 million available, on CPU.

**Top-1 accuracy: 13.5%** on held-out test data.

With 20,001 possible target words, random guessing gives 0.005%, so the model is roughly 2,700 times better than chance.

The main limits are the subsample size and vocabulary sparsity rather than the architecture. Training on the full set with a GPU is the obvious next step, and would be where I would start if continuing this.

## Requirements

```
pip install tensorflow pandas numpy scikit-learn flask
```

## Run order

```
python step1_data_cleaning.py
python step2_preprocessing.py
python step3_model.py
python step4_evaluate.py
```
