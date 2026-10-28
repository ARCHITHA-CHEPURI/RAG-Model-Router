from docx import Document

def extract_docx_text(file_path):
    doc = Document(file_path)

    text = "\n".join(p.text for p in doc.paragraphs)

    return {
        "filename": file_path,
        "type": "docx",
        "text": text
    }