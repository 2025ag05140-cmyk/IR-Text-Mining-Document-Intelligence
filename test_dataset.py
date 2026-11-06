
from modules.dataset import (
    load_dataset,
    validate_dataset,
    clean_dataset,
    get_corpus_statistics,
    save_dataset
)

# Load the raw BBC dataset
df = load_dataset("data/raw_dataset.csv")

# Validate
validate_dataset(df)

# Clean
cleaned_df = clean_dataset(df)

# Generate statistics
stats = get_corpus_statistics(cleaned_df)

print("\nCORPUS STATISTICS")
print("------------------------")

for key, value in stats.items():
    print(f"{key}: {value}")

# Save cleaned dataset
save_dataset(
    cleaned_df,
    "data/cleaned_dataset.csv"
)
