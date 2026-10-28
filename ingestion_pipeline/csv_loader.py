import pandas as pd


def extract_csv_text(file_path):
    df = pd.read_csv(file_path)

    rows = []

    for _, row in df.iterrows():

        sentence = []

        for column, value in row.items():
            sentence.append(f"{column}: {value}")

        rows.append(", ".join(sentence))

    text = "\n".join(rows)

    return {
        "filename": file_path,
        "type": "csv",
        "text": text
    }