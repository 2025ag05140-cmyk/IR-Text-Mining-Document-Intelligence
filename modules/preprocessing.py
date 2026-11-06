import re
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Local English stop-word list: no NLTK download required.
STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "is", "am", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "having", "do", "does", "did",
    "doing", "i", "me", "my", "myself", "we", "our", "ours", "you",
    "your", "yours", "he", "him", "his", "she", "her", "hers", "it",
    "its", "they", "them", "their", "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now",
    "could", "would", "may", "might", "must", "also"
}

STEMMER = PorterStemmer()
LEMMATIZER = WordNetLemmatizer()


def lowercase_text(text):
    return str(text).lower()


def tokenize(text):
    # Extract alphabetic words.
    return re.findall(r"\b[a-zA-Z]+\b", str(text).lower())


def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOP_WORDS]


def apply_stemming(tokens):
    return [STEMMER.stem(word) for word in tokens]


def apply_lemmatization(tokens):
    try:
        return [LEMMATIZER.lemmatize(word) for word in tokens]
    except LookupError as error:
        raise LookupError(
            "WordNet data is missing. Install the NLTK WordNet resource "
            "before running the lemmatization experiment."
        ) from error


def generate_ngrams(tokens, n=2):
    return [
        tuple(tokens[i:i + n])
        for i in range(len(tokens) - n + 1)
    ]


def preprocess_text(
    text,
    lowercase=True,
    remove_stopword=False,
    stemming=False,
    lemmatization=False
):
    if stemming and lemmatization:
        raise ValueError("Choose either stemming or lemmatization, not both.")

    if lowercase:
        text = lowercase_text(text)

    tokens = tokenize(text)

    if remove_stopword:
        tokens = remove_stopwords(tokens)

    if stemming:
        tokens = apply_stemming(tokens)
    elif lemmatization:
        tokens = apply_lemmatization(tokens)

    return tokens


def preprocess_documents(
    documents,
    lowercase=True,
    remove_stopword=False,
    stemming=False,
    lemmatization=False
):
    return [
        preprocess_text(
            doc,
            lowercase=lowercase,
            remove_stopword=remove_stopword,
            stemming=stemming,
            lemmatization=lemmatization
        )
        for doc in documents
    ]