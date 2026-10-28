from sentence_splitter import split_into_sentences

text = """
Artificial Intelligence is transforming healthcare.
Doctors are using AI to detect diseases.
Machine learning helps analyze medical images.
Deep learning models improve diagnostic accuracy.
"""

sentences = split_into_sentences(text)

for i, sentence in enumerate(sentences, start=1):
    print(f"{i}. {sentence}")