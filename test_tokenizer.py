# test_tokenizer.py - Check if your tokenizer is working

import pickle

print("Loading tokenizer...")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("\n" + "="*60)
print("TOKENIZER TEST")
print("="*60)

# Test 1: Check vocabulary size
print(f"\n✓ Total words in tokenizer: {len(tokenizer.word_index)}")

# Test 2: Show first 10 words
print("\nFirst 10 words in vocabulary:")
for i, (word, idx) in enumerate(list(tokenizer.word_index.items())[:10], 1):
    print(f"  {i}. {word:15} → ID {idx}")

# Test 3: Check if common Swedish words are there
print("\n" + "-"*60)
print("Testing COMMON SWEDISH WORDS:")
print("-"*60)

test_words = ["jag", "och", "i", "det", "är", "en", "som", "den", "på", "att"]
print("\nAre these words in the tokenizer?")
for word in test_words:
    word_id = tokenizer.word_index.get(word, 0)
    if word_id == 0:
        print(f"  ❌ '{word}' NOT FOUND")
    else:
        print(f"  ✅ '{word}' → ID {word_id}")

# Test 4: Try to encode some words
print("\n" + "-"*60)
print("ENCODING TEST - Convert words to numbers:")
print("-"*60)

test_sentence = "jag tycker om te och"
words = test_sentence.split()
print(f"\nSentence: {test_sentence}")
print(f"Words: {words}")

ids = [tokenizer.word_index.get(w, 0) for w in words]
print(f"IDs:   {ids}")

# Check if all are zeros (bad sign!)
if all(x == 0 for x in ids):
    print("\n❌ ERROR! All words got ID 0 (unknown)")
    print("   This means the tokenizer can't find ANY of these words!")
else:
    print(f"\n✅ Good! {sum(1 for x in ids if x != 0)}/{len(ids)} words were recognized")

# Test 5: Check vocabulary size vs max
print("\n" + "-"*60)
print("VOCABULARY SIZE:")
print("-"*60)

vocab_size = len(tokenizer.word_index)
print(f"\nTokenizer has {vocab_size:,} unique words")

# Check what MAX_VOCAB was used
if vocab_size == 20001:
    print("⚠️  Using 20k vocab (default)")
elif vocab_size == 50001:
    print("✅ Using 50k vocab (good!)")
elif vocab_size == 100001:
    print("✅ Using 100k vocab (excellent!)")
else:
    print(f"ℹ️  Using custom {vocab_size:,} vocab")

# Test 6: Decode (reverse lookup)
print("\n" + "-"*60)
print("REVERSE LOOKUP - Convert IDs back to words:")
print("-"*60)

id_to_word = {v: k for k, v in tokenizer.word_index.items()}
test_ids = [1, 2, 3, 4, 5]
print(f"\nIDs: {test_ids}")
decoded = [id_to_word.get(idx, "<UNK>") for idx in test_ids]
print(f"Words: {decoded}")

print("\n" + "="*60)
print("✓ TOKENIZER TEST COMPLETE")
print("="*60 + "\n")
