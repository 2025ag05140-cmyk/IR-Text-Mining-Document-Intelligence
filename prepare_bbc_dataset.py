
import pandas as pd
from pathlib import Path

DATASET_DIR = Path("data/bbc")
OUTPUT_FILE = Path("data/raw_dataset.csv")

CATEGORIES = [
    "business",
    "entertainment",
    "politics",
    "sport",
    "tech"
]


def prepare_bbc_dataset():

    records = []

    for category in CATEGORIES:

        folder = DATASET_DIR / category

        if not folder.exists():
            print(f"Missing folder: {category}")
            continue

        for file_path in sorted(folder.glob("*.txt")):

            try:
                content = file_path.read_text(
                    encoding="latin-1"
                ).strip()

                lines = [
                    line.strip()
                    for line in content.splitlines()
                    if line.strip()
                ]

                if len(lines) < 2:
                    print("Skipping:", file_path)
                    continue

                title = lines[0]
                article_text = "\n".join(lines[1:])

                records.append({
                    "document_id": f"{category}_{file_path.stem}",
                    "title": title,
                    "text": article_text,
                    "category": category
                })

            except Exception as error:
                print(f"Error reading {file_path}: {error}")

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No documents were loaded.")

    df = df.drop_duplicates(subset="document_id")
    df = df.dropna(subset=["title", "text", "category"])

    df = df[
        (df["title"].str.strip() != "") &
        (df["text"].str.strip() != "")
    ].copy()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print("\nDATASET PREPARATION COMPLETE")
    print("=" * 40)

    print("Total documents:", len(df))
    print("Total categories:", df["category"].nunique())

    print("\nCategory distribution:")
    print(df["category"].value_counts())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nAverage document length:")
    print(df["text"].str.split().str.len().mean())

    print("\nSaved to:", OUTPUT_FILE)

    return df


if __name__ == "__main__":
    df = prepare_bbc_dataset()
