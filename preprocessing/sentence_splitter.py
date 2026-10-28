import nltk
from nltk.tokenize import sent_tokenize


def split_into_sentences(text):
    """
    Splits text into meaningful sentences.
    """

    if not text:
        return []

    sentences = sent_tokenize(text)

    # Remove empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences