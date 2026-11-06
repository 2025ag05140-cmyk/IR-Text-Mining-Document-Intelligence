import json
from pathlib import Path

import pandas as pd

from modules.preprocessing import preprocess_documents
from modules.corpus_analytics import get_top_terms


INPUT_FILE = Path("data/raw_dataset.csv")
OUTPUT_DIR = Path("results/preprocessing_experiments")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_statistics(token_documents):
    """Calculate corpus and vocabulary statistics."""

    all_tokens = [
        token
        for document in token_documents
        for token in document
    ]

    vocabulary = set(all_tokens)

    document_lengths = [
        len(document)
        for document in token_documents
    ]

    return {
        "document_count": len(token_documents),
        "total_tokens": len(all_tokens),
        "vocabulary_size": len(vocabulary),
        "average_tokens_per_document": (
            sum(document_lengths) / len(document_lengths)
            if document_lengths else 0
        ),
        "minimum_document_length": (
            min(document_lengths) if document_lengths else 0
        ),
        "maximum_document_length": (
            max(document_lengths) if document_lengths else 0
        )
    }


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}. "
            "Check that data/raw_dataset.csv exists."
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = {"document_id", "title", "text", "category"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )

    # Use title + article text consistently in every experiment.
    documents = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    ).tolist()

    experiments = {
        "baseline": {
            "remove_stopword": False,
            "stemming": False,
            "lemmatization": False
        },
        "stopword_removal": {
            "remove_stopword": True,
            "stemming": False,
            "lemmatization": False
        },
        "stemming_only": {
            "remove_stopword": False,
            "stemming": True,
            "lemmatization": False
        },
        "lemmatization_only": {
            "remove_stopword": False,
            "stemming": False,
            "lemmatization": True
        }
    }

    summary_rows = []

    for experiment_name, settings in experiments.items():
        print(f"\nRunning experiment: {experiment_name}")

        token_documents = preprocess_documents(
            documents,
            lowercase=True,
            **settings
        )

        statistics = calculate_statistics(token_documents)

        statistics["experiment"] = experiment_name
        summary_rows.append(statistics)

        # Save the processed documents for this experiment.
        output_df = df.copy()

        output_df["tokens"] = [
            json.dumps(tokens)
            for tokens in token_documents
        ]

        output_df["processed_text"] = [
            " ".join(tokens)
            for tokens in token_documents
        ]

        output_df.to_csv(
            OUTPUT_DIR / f"{experiment_name}_dataset.csv",
            index=False
        )

        # Save the 20 most frequent terms.
        top_terms = get_top_terms(token_documents, top_n=20)

        if isinstance(top_terms, dict):
            top_terms_df = pd.DataFrame(
                top_terms.items(),
                columns=["term", "frequency"]
            )
        else:
            top_terms_df = pd.DataFrame(
                top_terms,
                columns=["term", "frequency"]
            )

        top_terms_df.to_csv(
            OUTPUT_DIR / f"{experiment_name}_top_terms.csv",
            index=False
        )

        print(statistics)

    # Save a comparison table for all experiments.
    summary_df = pd.DataFrame(summary_rows)

    summary_df = summary_df[
        [
            "experiment",
            "document_count",
            "total_tokens",
            "vocabulary_size",
            "average_tokens_per_document",
            "minimum_document_length",
            "maximum_document_length"
        ]
    ]

    summary_df.to_csv(
        OUTPUT_DIR / "experiment_comparison.csv",
        index=False
    )

    print("\nAll experiments completed!")
    print(summary_df.to_string(index=False))
    print(f"\nResults saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()