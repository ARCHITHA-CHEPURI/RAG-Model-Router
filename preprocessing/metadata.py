import os


def generate_metadata(doc, chunk_number, total_chunks, text):
    """
    Generate metadata for a chunk.
    """

    filename = doc.get("filename", "")
    source = doc.get("source", "")
    document_type = doc.get("document_type", doc.get("type", ""))

    metadata = {
        "filename": os.path.basename(filename),
        "document_name": os.path.basename(filename),
        "source": source,
        "document_type": document_type,
        "chunk_number": chunk_number,
        "total_chunks": total_chunks,
        "word_count": len(text.split()),
        "character_count": len(text)
    }

    return metadata