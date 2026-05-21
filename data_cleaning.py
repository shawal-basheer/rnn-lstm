"""
=============================================================
  STEP 1 — DATA LOADING & CLEANING
  RNN / LSTM Next-Word Prediction  |  DT086A Machine Learning
=============================================================

WHAT THIS STEP DOES
-------------------
1. Load the raw CSV dataset (Swedish news texts)
2. Clean every article body:
     • Strip HTML tags          (e.g.  <p class="body">  →  "")
     • Remove special / junk characters   (keep only letters, digits,
       Swedish åäö, spaces, and basic punctuation . ? !)
     • Collapse extra whitespace
     • Filter out empty / very short texts
3. Save the cleaned text to a plain .txt file so later steps
   can read it without repeating all this work.

WHY EACH DECISION WAS MADE
---------------------------
- We work with the 'regexp_replace' column (the article body),
  NOT the headline, because it contains full sentences needed
  to build meaningful 5-word → 6th-word training pairs.
- HTML stripping comes first; otherwise regex word splits
  would treat tag attributes as words.
- We lower-case everything so "Sverige" and "sverige" are
  the same token — this keeps the vocabulary smaller.
- Very short texts (< 10 words) are discarded because they
  cannot form even a single 5+1 training sequence.
"""

import re
import pandas as pd

# 1. Load raw data
CSV_PATH = "../text-ds.csv"          # adjust if you run from a different folder
OUTPUT_PATH = "cleaned_corpus.txt"

print("Loading dataset …")
df = pd.read_csv(CSV_PATH)
print(f"  Loaded {len(df):,} rows  |  columns: {df.columns.tolist()}")

# 2. Pick the correct column
# 'regexp_replace' holds the full article HTML; 'headline' is just the title.
raw_texts = df["regexp_replace"].dropna().tolist()
print(f"  Non-null article bodies: {len(raw_texts):,}")

# 3. Define a cleaning pipeline

def clean_text(text: str) -> str:
    """
    Apply a pipeline of cleaning steps to one article string.

    Steps
    -----
    a) Strip HTML tags          <p class="body">Hello</p>  →  Hello
    b) Decode common HTML entities  &amp; → &,  &gt; → >  etc.
    c) Remove URLs              https://... → ""
    d) Keep only allowed chars  [a-zåäöA-ZÅÄÖ0-9 .,?!'-]
    e) Lower-case everything
    f) Collapse multiple spaces / newlines → single space
    g) Strip leading / trailing whitespace
    """
    # a) Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # b) Common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&gt;", ">")
    text = text.replace("&lt;", "<")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&#160;", " ")

    # c) Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # d) Keep letters (incl. Swedish), digits, spaces, and . ? ! ' -
    #    Everything else (brackets, @, #, etc.) is replaced with a space.
    text = re.sub(r"[^a-zA-ZåäöÅÄÖ0-9\s.,?!\'-]", " ", text)

    # e) Lower-case
    text = text.lower()

    # f) Collapse whitespace (spaces, tabs, newlines) → single space
    text = re.sub(r"\s+", " ", text)

    # g) Strip edges
    text = text.strip()

    return text


# 4. Run the pipeline on every article
print("Cleaning texts …")
cleaned = []
skipped = 0

for raw in raw_texts:
    c = clean_text(raw)
    word_count = len(c.split())
    if word_count >= 10:          # need at least 10 words to be useful
        cleaned.append(c)
    else:
        skipped += 1

print(f"  Kept   : {len(cleaned):,} articles")
print(f"  Skipped: {skipped:,}  (too short after cleaning)")

# 5. Quick peek at the cleaned data 
print("\n── Sample cleaned article (first 300 chars) ──")
print(cleaned[0][:300])
print("\n── Sample cleaned article (second entry, first 300 chars) ──")
print(cleaned[1][:300])

# 6. Save to file
# Each article on its own line — easy to read back later.
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for article in cleaned:
        f.write(article + "\n")

print(f"\nSaved {len(cleaned):,} cleaned articles → '{OUTPUT_PATH}'")

# 7. Basic stats
all_words = " ".join(cleaned).split()
unique_words = set(all_words)

print("\n── Corpus statistics ──")
print(f"  Total tokens (words)  : {len(all_words):,}")
print(f"  Unique tokens         : {len(unique_words):,}")
print(f"  Average article length: {len(all_words) // len(cleaned):,} words")
print("\nStep 1 complete ✓")
