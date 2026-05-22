# Step 4 - Evaluation and Next-Word Prediction
#
# This is the final step. I load the trained model and check how
# well it actually performs on data it has never seen before
# (the test set from step 2).
#
# I also build a small prediction function that takes 5 words as
# input and outputs the model's top-3 guesses for the next word.
#
# Evaluation metrics I use:
#
#   Accuracy       - what % of the time does the model pick exactly
#                    the correct word as its top-1 prediction.
#
#   Top-5 accuracy - what % of the time is the correct word anywhere
#                    in the model's top 5 predictions.
#                    This is a fairer measure because language is
#                    flexible - multiple words could follow a sequence.
#
#   Loss           - how confident and correct the model is overall.
#                    Lower = better.
#
# Why accuracy looks "low" but is actually good:
#   With 20 001 possible words, random guessing = 0.005% accuracy.
#   Even 6-7% means the model is ~1400x better than random.
#   More training epochs and more data would push this higher.

import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ------------------------------------------------------------------
# 1. Load everything saved from previous steps
# ------------------------------------------------------------------
print("Loading model and data ...")

model = load_model("best_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("vocab_size.txt", "r") as f:
    vocab_size = int(f.read().strip())

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Build reverse lookup: ID → word (needed to decode predictions)
id_to_word = {idx: word for word, idx in tokenizer.word_index.items()}

print(f"  Model loaded     : best_model.keras")
print(f"  Test sequences   : {len(X_test):,}")
print(f"  Vocab size       : {vocab_size:,}")

# ------------------------------------------------------------------
# 2. Evaluate on test set
#
# I use a subsample of the test set (10 000 sequences) to keep
# evaluation fast. The test set was never seen during training so
# these results are a fair reflection of real performance.
# ------------------------------------------------------------------
print("\nEvaluating on test set ...")

np.random.seed(42)
idx = np.random.choice(len(X_test), min(10_000, len(X_test)), replace=False)
X_eval = X_test[idx]
y_eval = y_test[idx]

loss, accuracy = model.evaluate(X_eval, y_eval, batch_size=512, verbose=0)

print(f"\n── Test set results ──")
print(f"  Loss          : {loss:.4f}")
print(f"  Top-1 Accuracy: {accuracy*100:.2f}%")

# ------------------------------------------------------------------
# 3. Top-5 accuracy
#
# For each test sequence, I get the model's full probability
# distribution over all 20 001 words, then check if the correct
# answer is in the top 5 highest probabilities.
# ------------------------------------------------------------------
print("\nCalculating top-5 accuracy ...")

predictions = model.predict(X_eval, batch_size=512, verbose=0)
top5_correct = 0

for i in range(len(X_eval)):
    top5_ids = np.argsort(predictions[i])[-5:]   # indices of top 5 probs
    if y_eval[i] in top5_ids:
        top5_correct += 1

top5_accuracy = top5_correct / len(X_eval)
print(f"  Top-5 Accuracy: {top5_accuracy*100:.2f}%")

# ------------------------------------------------------------------
# 4. Training history summary
#
# Load the history saved in step 3 and print a clean summary.
# If you trained more than 1 epoch this shows the progression.
# ------------------------------------------------------------------
history = np.load("training_history.npy", allow_pickle=True).item()

print(f"\n── Training history ──")
num_epochs = len(history["accuracy"])
print(f"  Epochs trained : {num_epochs}")
for ep in range(num_epochs):
    print(f"  Epoch {ep+1:02d}  |  "
          f"train_acc: {history['accuracy'][ep]*100:.2f}%  |  "
          f"val_acc: {history['val_accuracy'][ep]*100:.2f}%  |  "
          f"val_loss: {history['val_loss'][ep]:.4f}")

# ------------------------------------------------------------------
# 5. Next-word prediction function
#
# This is the actual output of the whole project -
# give it 5 Swedish words, get back the model's top-3 predictions
# for what word comes next.
#
# temperature controls creativity:
#   low  (e.g. 0.5) = model picks safe/common words
#   high (e.g. 1.5) = model picks more surprising words
# ------------------------------------------------------------------

def predict_next_word(input_words: list, top_n: int = 3, temperature: float = 1.0):
    """
    Given a list of 5 words, predict the most likely next word.

    Parameters
    ----------
    input_words : list of 5 strings (Swedish)
    top_n       : how many top predictions to return
    temperature : float, controls randomness (1.0 = no change)

    Returns
    -------
    list of (word, probability%) tuples, highest first
    """
    if len(input_words) != 5:
        raise ValueError("Please provide exactly 5 words.")

    # Convert words to their integer IDs
    # Unknown words get ID 0 (OOV token)
    token_ids = []
    for w in input_words:
        idx = tokenizer.word_index.get(w.lower(), 0)
        token_ids.append(idx)

    # Reshape to (1, 5) - model expects a batch
    X = np.array([token_ids])

    # Get probability distribution over all words
    probs = model.predict(X, verbose=0)[0]

    # Apply temperature scaling
    # dividing by temperature < 1 makes the distribution sharper
    # (model becomes more confident about its top choice)
    probs = np.log(probs + 1e-10) / temperature
    probs = np.exp(probs)
    probs = probs / probs.sum()

    # Pick the top_n highest probability words
    top_indices = np.argsort(probs)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        word = id_to_word.get(idx, "<unknown>")
        prob = probs[idx] * 100
        results.append((word, round(float(prob), 2)))

    return results


# ------------------------------------------------------------------
# 6. Demo predictions
#
# Try a few Swedish 5-word sequences and see what the model predicts.
# After only 1 epoch on 50k sequences these won't be perfect but
# you should see the model making somewhat sensible guesses.
# ------------------------------------------------------------------
print("\n── Live next-word predictions ──")

test_sequences = [
    ["det", "är", "en", "stor", "fråga"],
    ["polisen", "söker", "efter", "en", "man"],
    ["kommunen", "har", "beslutat", "att", "bygga"],
    ["priset", "på", "bostäder", "i", "stockholm"],
    ["det", "nya", "systemet", "ska", "införas"],
]

for words in test_sequences:
    preds = predict_next_word(words, top_n=3)
    print(f"\n  Input : '{' '.join(words)}'")
    print(f"  Top 3 predictions:")
    for rank, (word, prob) in enumerate(preds, 1):
        bar = "█" * int(prob / 2)
        print(f"    {rank}. {word:<20} {prob:5.2f}%  {bar}")

# ------------------------------------------------------------------
# 7. Interactive mode
#    If you run this script directly it lets you type your own words
# ------------------------------------------------------------------
print("\n── Try it yourself ──")
print("Type 5 Swedish words separated by spaces (or press Enter to skip):")

try:
    user_input = input("> ").strip()
    if user_input:
        words = user_input.split()
        if len(words) == 5:
            preds = predict_next_word(words, top_n=5)
            print(f"\nTop 5 predictions after '{' '.join(words)}':")
            for rank, (word, prob) in enumerate(preds, 1):
                print(f"  {rank}. {word:<20} {prob:.2f}%")
        else:
            print(f"  Got {len(words)} words, need exactly 5. Skipping.")
except (EOFError, KeyboardInterrupt):
    pass  # non-interactive environment, skip

print("\nStep 4 done ✓")
print("Project complete! All 4 steps finished.")
