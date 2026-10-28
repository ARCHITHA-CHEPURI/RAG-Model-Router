import os
import sys
import json

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import config
from csv_loader import extract_csv_text
from docs_loader import extract_docx_text
from pdfs_loader import extract_pdf_text
from ppt_loader import extract_ppt_text

# Path to DATA folder
DATA_FOLDER = config.DATA_DIR

print("DATA Folder:", DATA_FOLDER)
print("Exists:", os.path.exists(DATA_FOLDER))

documents = []

for root, dirs, files in os.walk(DATA_FOLDER):
    # Exclude processed and vector_db directories from scan
    dirs[:] = [d for d in dirs if d.lower() not in ["processed", "vector_db"]]
    print(f"\nScanning Folder: {root}")

    for file in files:
        path = os.path.join(root, file)
        print("Found:", file)

        try:
            if file.lower().endswith(".pdf"):
                documents.append(extract_pdf_text(path))

            elif file.lower().endswith(".docx"):
                documents.append(extract_docx_text(path))

            elif file.lower().endswith(".pptx"):
                documents.append(extract_ppt_text(path))

            elif file.lower().endswith(".csv"):
                documents.append(extract_csv_text(path))

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Save JSON inside project processed folder
output_path = config.PROCESSED_DOCUMENTS
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=4, ensure_ascii=False)

print("\n--------------------------------")
print(f"Processed {len(documents)} documents.")
print(f"Saved to: {output_path}")
print("--------------------------------")