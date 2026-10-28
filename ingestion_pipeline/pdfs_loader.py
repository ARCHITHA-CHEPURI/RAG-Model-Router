import fitz

def extract_pdf_text(file_path):
    doc = fitz.open(file_path)

    text = ""

    for page in doc:
        text += page.get_text()

    return {
        "filename": file_path,
        "type": "pdf",
        "text": text
    }