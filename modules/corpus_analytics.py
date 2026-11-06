
from collections import Counter


def get_vocabulary(documents):
    """Return all unique terms across documents."""

    vocabulary = set()

    for tokens in documents:
        vocabulary.update(tokens)

    return vocabulary


def get_term_frequencies(documents):
    """Calculate frequency of each term."""

    frequencies = Counter()

    for tokens in documents:
        frequencies.update(tokens)

    return frequencies


def get_document_lengths(documents):
    """Calculate number of tokens in each document."""

    return [
        len(tokens)
        for tokens in documents
    ]


def get_top_terms(documents, top_n=20):
    """Return the most frequent terms."""

    frequencies = get_term_frequencies(documents)

    return frequencies.most_common(top_n)


def summarize_tokens(documents):
    """Calculate corpus-level token statistics."""

    lengths = get_document_lengths(documents)

    vocabulary = get_vocabulary(documents)

    total_terms = sum(lengths)

    if not lengths:
        return {
            "total_documents": 0,
            "vocabulary_size": 0,
            "total_terms": 0,
            "average_document_length": 0,
            "minimum_document_length": 0,
            "maximum_document_length": 0
        }

    return {
        "total_documents": len(documents),
        "vocabulary_size": len(vocabulary),
        "total_terms": total_terms,
        "average_document_length": round(
            total_terms / len(documents), 2
        ),
        "minimum_document_length": min(lengths),
        "maximum_document_length": max(lengths)
    }
