
import json
import pandas as pd

from modules.dataset import (
    load_dataset,
    validate_dataset,
    clean_dataset,
    get_corpus_statistics
)

from modules.preprocessing import preprocess_documents

from modules.corpus_analytics import (
    summarize_tokens,
    get_top_terms
)

from modules.duplicates_detection import (
    exact_duplicates,
    near_duplicates
)


# 1. Load and clean dataset
df = load_dataset("data/raw_dataset.csv")

validate_dataset(df)

df = clean_dataset(df)


# 2. Statistics before preprocessing
original_tokens = preprocess_documents(
    df["text"].tolist(),
    lowercase=True,
    remove_stopword=False
)

before_stats = summarize_tokens(original_tokens)


# 3. Apply configurable preprocessing
processed_tokens = preprocess_documents(
    df["text"].tolist(),
    lowercase=True,
    remove_stopword=True,
    stemming=False,
    lemmatization=True
)

after_stats = summarize_tokens(processed_tokens)


# 4. Add processed tokens
df["tokens"] = [
    json.dumps(tokens)
    for tokens in processed_tokens
]

df["processed_text"] = [
    " ".join(tokens)
    for tokens in processed_tokens
]


# 5. Duplicate analysis
exact = exact_duplicates(df["text"].tolist())

near = near_duplicates(
    processed_tokens,
    threshold=0.8
)


# 6. Save processed dataset
df.to_csv(
    "data/processed_dataset.csv",
    index=False,
    encoding="utf-8"
)


# 7. Display results
print("\nBEFORE PREPROCESSING")
print(before_stats)

print("\nAFTER PREPROCESSING")
print(after_stats)

print("\nEXACT DUPLICATE GROUPS:", len(exact))
print("NEAR DUPLICATE PAIRS:", len(near))

print("\nTOP 20 TERMS")
print(get_top_terms(processed_tokens, 20))

print("\nPROCESSED DATASET SAVED!")
