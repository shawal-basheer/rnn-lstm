# Step 2 - Preprocessing the cleaned text data
# Here I take the cleaned articles from step 1 and turn them into
# something the LSTM model can actually train on.
#
# The idea is simple:
#   - every 5 words in a row = input (X)
#   - the word right after those 5 = what we want to predict (y)
#
# So for the sentence "det är en riktig mastodonträttegång som inleds"
#   X = [det, är, en, riktig, mastodonträttegång]
#   y = som
#
# Then the window slides one word forward:
#   X = [är, en, riktig, mastodonträttegång, som]
#   y = inleds
# ...and so on for every article.
#
# NOTE on labels:
#   I store y as plain integer IDs (sparse), NOT one-hot vectors.
#   One-hot would mean a vector of 20 000 zeros per sample which
#   would need ~587 GB of RAM for 4M sequences - not realistic.
#   Instead I use sparse_categorical_crossentropy loss in the model
#   which accepts integer labels directly. Same math, 0 extra memory.

import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

CORPUS_FILE  = "cleaned_corpus.txt"
SEQUENCE_LEN = 5      # how many words we feed in as input
MAX_VOCAB    = 20000  # only keep the 20k most common words (keeps model manageable)

# 1. Read the cleaned corpus back in

print("Reading cleaned corpus ...")
with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    articles = f.readlines()

articles = [a.strip() for a in articles if a.strip()]
print(f"  Articles loaded: {len(articles):,}")

# 2. Tokenize - give every unique word its own integer ID
#
# Why do we do this?
#   Neural networks work with numbers, not strings.
#   Tokenizer builds a dictionary like:
#     { "det": 1, "är": 2, "en": 3, ... }
#   and then converts every word in the corpus to its ID.
#
# num_words=MAX_VOCAB means only the 20 000 most frequent words
# get their own ID. Rare words (typos, one-off names) are ignored.
# This keeps the vocabulary - and our model - a sensible size.

print("\nFitting tokenizer on corpus ...")
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(articles)

vocab_size = min(MAX_VOCAB, len(tokenizer.word_index)) + 1
print(f"  Total unique words in corpus : {len(tokenizer.word_index):,}")
print(f"  Vocabulary size we will use  : {vocab_size:,}")

# 3. Build sequences
#
# For each article I slide a window of 6 words across the text.
# First 5 words → X (input), 6th word → y (label/target).

print("\nBuilding sequences ...")
X_raw = []
y_raw = []

for article in articles:
    # Convert the article text to a list of integer IDs
    token_ids = tokenizer.texts_to_sequences([article])[0]

    # Slide the window
    for i in range(len(token_ids) - SEQUENCE_LEN):
        seq_in  = token_ids[i : i + SEQUENCE_LEN]       # 5 words
        seq_out = token_ids[i + SEQUENCE_LEN]            # 6th word

        # Skip if any token got mapped to 0 (unknown / out of vocab)
        if 0 not in seq_in and seq_out != 0:
            X_raw.append(seq_in)
            y_raw.append(seq_out)

print(f"  Total sequences created: {len(X_raw):,}")

# 4. Convert to numpy arrays
#
# X shape: (num_sequences, 5)  - each row is 5 word IDs
# y shape: (num_sequences,)    - one integer per sequence, the target word ID
#
# I keep y as plain integers rather than one-hot vectors, for the
# memory reason explained at the top of this file. The model uses
# sparse_categorical_crossentropy, which takes integer labels directly.

print("\nConverting to numpy arrays ...")
X = np.array(X_raw, dtype=np.int32)
y = np.array(y_raw, dtype=np.int32)  # sparse integer IDs, not one-hot

print(f"  X shape: {X.shape}  (sequences × window_size)")
print(f'  y shape: {y.shape}  (one integer per sequence = target word ID)')

# 5. Train / Validation / Test split
#
# I split the data into three parts:
#   80% train      - model learns from this
#   10% validation - checked after each epoch to catch overfitting
#   10% test       - final evaluation, model never sees this during training
#
# random_state=42 just makes the split reproducible every time I run it.

print("\nSplitting into train / val / test ...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"  Train      : {X_train.shape[0]:>10,} sequences")
print(f"  Validation : {X_val.shape[0]:>10,} sequences")
print(f"  Test       : {X_test.shape[0]:>10,} sequences")

# 6. Save everything so step 3 can just load it
#
# I save the tokenizer as a pickle file so I don't have to rebuild
# the vocabulary from scratch every time I train or run a prediction.

print("\nSaving preprocessed data ...")

np.save("X_train.npy", X_train)
np.save("X_val.npy",   X_val)
np.save("X_test.npy",  X_test)
np.save("y_train.npy", y_train)
np.save("y_val.npy",   y_val)
np.save("y_test.npy",  y_test)

with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Also save vocab_size so step 3 knows the output layer size
with open("vocab_size.txt", "w") as f:
    f.write(str(vocab_size))

print("  Saved: X_train/val/test.npy")
print("  Saved: y_train/val/test.npy")
print("  Saved: tokenizer.pkl")
print("  Saved: vocab_size.txt")

# 7. Quick sanity check - decode a sequence back to words

# y is stored as a plain integer word ID, so I can look it up
# directly in the reverse dictionary.

print("\n── Sanity check: decoding first sequence back to words ──")
id_to_word = {v: k for k, v in tokenizer.word_index.items()}

sample_X = X_train[0]
sample_y = y_train[0]

input_words  = [id_to_word.get(idx, "?") for idx in sample_X]
target_word  = id_to_word.get(sample_y, "?")

print(f"  Input  (5 words) : {' '.join(input_words)}")
print(f"  Target (6th word): {target_word}")

print("\nStep 2 done ✓")