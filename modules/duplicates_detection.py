
import hashlib
from itertools import combinations


def exact_duplicates(documents):
    """
    Identify exact duplicates after basic normalization.
    Returns groups of document indices.
    """

    hash_groups = {}

    for index, text in enumerate(documents):

        normalized = " ".join(
            str(text).lower().split()
        )

        digest = hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

        hash_groups.setdefault(
            digest, []
        ).append(index)

    return [
        indices
        for indices in hash_groups.values()
        if len(indices) > 1
    ]


def jaccard_similarity(tokens_a, tokens_b):
    """Calculate Jaccard similarity between two token sets."""

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    union = set_a.union(set_b)

    if not union:
        return 1.0

    intersection = set_a.intersection(set_b)

    return len(intersection) / len(union)


def near_duplicates(
    token_documents,
    threshold=0.8
):
    """
    Identify near-duplicate document pairs.
    """

    results = []

    for i, j in combinations(
        range(len(token_documents)), 2
    ):

        similarity = jaccard_similarity(
            token_documents[i],
            token_documents[j]
        )

        if similarity >= threshold:
            results.append({
                "document_index_1": i,
                "document_index_2": j,
                "similarity": round(similarity, 4)
            })

    return results
