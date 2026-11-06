
from modules.preprocessing import (
    preprocess_text,
    generate_ngrams
)

sample_text = """
The companies are developing innovative technologies
for better healthcare solutions.
"""

# Original tokenization
tokens = preprocess_text(
    sample_text,
    lowercase=True,
    remove_stopword=False
)

print("Original tokens:")
print(tokens)

# Stop-word removal
cleaned = preprocess_text(
    sample_text,
    lowercase=True,
    remove_stopword=True
)

print("\nAfter stop-word removal:")
print(cleaned)

# Stemming
stemmed = preprocess_text(
    sample_text,
    remove_stopword=True,
    stemming=True
)

print("\nAfter stemming:")
print(stemmed)

# Lemmatization
lemmatized = preprocess_text(
    sample_text,
    remove_stopword=True,
    lemmatization=True
)

print("\nAfter lemmatization:")
print(lemmatized)

# Bigrams
bigrams = generate_ngrams(cleaned, n=2)

print("\nBigrams:")
print(bigrams)

# Trigrams
trigrams = generate_ngrams(cleaned, n=3)

print("\nTrigrams:")
print(trigrams)
