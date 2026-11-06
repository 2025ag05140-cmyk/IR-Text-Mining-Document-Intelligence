
import json
from pathlib import Path

import pandas as pd

from modules.preprocessing import preprocess_documents


INPUT_FILE = Path("data/raw_dataset.csv")
OUTPUT_FILE = Path("data/processed_dataset.csv")


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "document_id",
        "title",
        "text",
        "category"
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()

    for column in required_columns:
        df[column] = df[column].fillna("").astype(str).str.strip()

    df = df[
        (df["document_id"] != "")
        & ((df["title"] != "") | (df["text"] != ""))
        & (df["category"] != "")
    ].copy()

    df = df.drop_duplicates(
        subset=["document_id"]
    ).reset_index(drop=True)

    documents = (
        df["title"] + " " + df["text"]
    ).tolist()

    tokens = preprocess_documents(
        documents,
        lowercase=True,
        remove_stopword=True,
        stemming=False,
        lemmatization=False
    )

    df["tokens"] = [
        json.dumps(document_tokens)
        for document_tokens in tokens
    ]

    df["processed_text"] = [
        " ".join(document_tokens)
        for document_tokens in tokens
    ]

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("Shared dataset created successfully!")
    print("Output:", OUTPUT_FILE)
    print("Documents:", len(df))
    print("Unique IDs:", df["document_id"].nunique())
    print(
        "Vocabulary size:",
        len({
            token
            for document in tokens
            for token in document
        })
    )

    print("\nColumns:", df.columns.tolist())
    print("\nCategory distribution:")
    print(df["category"].value_counts())


if __name__ == "__main__":
    main()
