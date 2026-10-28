import re
import unicodedata


def clean_text(text):
    """
    Cleans extracted text from PDFs, DOCX, PPTX and CSV.
    """

    if not text:
        return ""

    # -----------------------------
    # Unicode normalization
    # -----------------------------
    text = unicodedata.normalize("NFKC", text)

    # -----------------------------
    # Replace tabs
    # -----------------------------
    text = text.replace("\t", " ")

    # -----------------------------
    # Remove URLs
    # -----------------------------
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\.\S+", "", text)

    # -----------------------------
    # Remove email addresses
    # -----------------------------
    text = re.sub(r"\S+@\S+", "", text)

    # -----------------------------
    # Remove citation markers
    # Example:
    # [12]
    # (12)
    # -----------------------------
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\(\d+\)", "", text)

    # -----------------------------
    # Remove page numbers
    # -----------------------------
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # -----------------------------
    # Remove standalone numbers
    # -----------------------------
    text = re.sub(r"^\d+$", "", text, flags=re.MULTILINE)

    # -----------------------------
    # Remove "REFERENCES"
    # -----------------------------
    text = re.sub(r"\bREFERENCES\b", "", text, flags=re.IGNORECASE)

    # -----------------------------
    # Remove repeated punctuation
    # -----------------------------
    text = re.sub(r"([.,!?]){2,}", r"\1", text)

    # -----------------------------
    # Join broken PDF lines
    # -----------------------------
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # -----------------------------
    # Normalize spaces
    # -----------------------------
    text = re.sub(r"[ ]+", " ", text)

    # -----------------------------
    # Normalize blank lines
    # -----------------------------
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()