
import pandas as pd
from pathlib import Path
import json


REQUIRED_COLUMNS = [
    "document_id",
    "title",
    "text",
    "category"
]


def load_dataset(file):
    """
    Load a dataset from CSV or JSON.
    Supports file paths and Streamlit uploaded files.
    """

    filename = getattr(file, "name", str(file)).lower()

    if filename.endswith(".csv"):
        return pd.read_csv(file)

    elif filename.endswith(".json"):
        data = json.load(file)

        if isinstance(data, dict):
            data = data.get("data", [])

        return pd.DataFrame(data)

    else:
        raise ValueError(
            "Unsupported file format. Upload CSV or JSON."
        )


def validate_dataset(df):
    """
    Validate the dataset structure.
    """

    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    return True


def clean_dataset(df):
    """
    Clean the dataset while preserving original text.
    """

    df = df.copy()

    # Standardize document IDs
    df["document_id"] = (
        df["document_id"]
        .astype("string")
        .str.strip()
    )

    # Standardize text metadata
    for col in ["title", "text", "category"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    # Remove invalid records
    df = df[
        (df["document_id"] != "") &
        (df["title"] != "") &
        (df["text"] != "") &
        (df["category"] != "")
    ].copy()

    # Remove duplicate document IDs
    duplicate_ids = int(
        df["document_id"].duplicated().sum()
    )

    df = df.drop_duplicates(
        subset=["document_id"],
        keep="first"
    )

    df = df.reset_index(drop=True)

    print("Dataset cleaning complete")
    print("Duplicate IDs removed:", duplicate_ids)
    print("Remaining documents:", len(df))

    return df


def get_corpus_statistics(df):
    """
    Calculate basic corpus statistics.
    """

    document_lengths = (
        df["text"].astype(str).str.split().str.len()
    )

    return {
        "total_documents": len(df),
        "total_categories": df["category"].nunique(),
        "total_words": int(document_lengths.sum()),
        "average_document_length": round(
            document_lengths.mean(), 2
        ),
        "minimum_document_length": int(
            document_lengths.min()
        ),
        "maximum_document_length": int(
            document_lengths.max()
        ),
        "category_distribution": (
            df["category"].value_counts().to_dict()
        ),
        "missing_values": int(df.isnull().sum().sum())
    }


def save_dataset(df, output_path):
    """
    Save cleaned dataset to CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8"
    )

    print("Dataset saved:", output_path)
