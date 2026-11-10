# Step 3 - Building and Training the LSTM Model
#
# Now that I have the sequences ready from step 2, I can build the
# actual neural network and train it.
#
# The model stacks 4 layers on top of each other:
#
#   [Embedding]   word ID (a number) → dense vector of 100 numbers
#                 Similar words end up with similar vectors.
#                 Learned automatically during training.
#
#   [LSTM]        Reads the 5 word vectors one by one, keeps a
#                 "memory" of what it has seen. This is the core
#                 recurrent part that understands word order.
#
#   [Dense 64]    A standard fully-connected layer with ReLU.
#                 Helps learn more complex patterns.
#
#   [Dropout 0.5] Randomly switches off 50% of neurons each step.
#                 Forces the model not to rely on any single neuron.
#                 Prevents overfitting (= memorising the data).
#
#   [Dense output] One neuron per word in the vocabulary (20 001).
#                  Softmax turns them into probabilities.
#                  Highest probability = predicted next word.
#
# NOTE on dataset size:
#   The full training set has 3.1 million sequences which would take
#   hours on a CPU. I subsample 50 000 sequences here - this is a
#   standard practice called subsampling and still shows the model
#   learning. On a GPU with more time, you would train on all of it.

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ------------------------------------------------------------------
# 1. Load data from step 2
# ------------------------------------------------------------------
print("Loading preprocessed data ...")
X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
y_train = np.load("y_train.npy")
y_val   = np.load("y_val.npy")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("vocab_size.txt", "r") as f:
    vocab_size = int(f.read().strip())

SEQUENCE_LEN = X_train.shape[1]
print(f"  Vocab size   : {vocab_size:,}")
print(f"  Sequence len : {SEQUENCE_LEN}")
print(f"  Full train   : {len(X_train):,} sequences")

# ------------------------------------------------------------------
# 2. Subsample to make training feasible on a CPU
#    (remove this block if you have a GPU and more time)
# ------------------------------------------------------------------
SAMPLE_SIZE     = 500_000
SAMPLE_SIZE_VAL = 50_000

np.random.seed(42)
idx = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
X_tr = X_train[idx]
y_tr = y_train[idx]

idx_v = np.random.choice(len(X_val), SAMPLE_SIZE_VAL, replace=False)
X_v = X_val[idx_v]
y_v = y_val[idx_v]

print(f"\n  Training on subsample : {len(X_tr):,}")
print(f"  Validating on         : {len(X_v):,}")

# ------------------------------------------------------------------
# 3. Build the model
# ------------------------------------------------------------------
print("\nBuilding model ...")

model = Sequential([
    # Embedding: vocab_size words → 100-dim vectors
    Embedding(input_dim=vocab_size, output_dim=100),

    # LSTM: 64 memory units, small internal dropout for regularisation
    LSTM(64, return_sequences=False, dropout=0.1, recurrent_dropout=0.1),

    # Hidden layer
    Dense(64, activation="relu"),

    # Dropout before output
    Dropout(0.5),

    # Output: probability over every word in vocabulary
    Dense(vocab_size, activation="softmax")
])

# sparse_categorical_crossentropy works directly with integer labels
# so I don't need to one-hot encode y - saves a huge amount of memory
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------------------
# 4. Callbacks
#
# EarlyStopping  - stops training if val_loss does not improve
#                  for 3 epochs in a row. Saves time + avoids
#                  overfitting.
#
# ModelCheckpoint - saves the model every time val_loss improves.
#                   So even if I stop training early I keep the
#                   best version found so far.
# ------------------------------------------------------------------
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath="best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
]

# ------------------------------------------------------------------
# 5. Train
# ------------------------------------------------------------------
print("\nStarting training ...")
BATCH_SIZE = 512   # sequences processed before each weight update
EPOCHS     = 20    # max passes through the training data

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_v, y_v),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ------------------------------------------------------------------
# 6. Save history for plotting in step 4
# ------------------------------------------------------------------
np.save("training_history.npy", history.history)

best_epoch   = int(np.argmin(history.history["val_loss"])) + 1
best_val_acc = history.history["val_accuracy"][best_epoch - 1]

print("\n── Training complete ──")
print(f"  Best epoch       : {best_epoch}")
print(f"  Best val accuracy: {best_val_acc*100:.2f}%")
print(f"  Model saved      : best_model.keras")
print("\nStep 3 done ✓")
