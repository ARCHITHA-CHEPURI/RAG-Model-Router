import numpy as np
from sentence_splitter import split_into_sentences


class SemanticChunker:
    """
    Advanced chunker supporting:
    1. Recursive Character splitting (default) - respects structural boundaries.
    2. Semantic Similarity splitting - uses embedding shifts between sentences.
    3. Naive Sentence Sliding Window - for backward compatibility.
    """

    def __init__(
        self,
        method="recursive",
        max_words=450,
        overlap_words=50,
        overlap_sentences=2,
        embedding_function=None,
        similarity_threshold=0.6,
    ):
        self.method = method
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.overlap_sentences = overlap_sentences
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold

    def chunk_text(self, text):
        if not text or not text.strip():
            return []

        if self.method == "semantic" and self.embedding_function is not None:
            return self._chunk_semantically(text)
        elif self.method == "sentence_sliding":
            return self._chunk_sentence_sliding(text)
        else:
            # Default to recursive character splitting
            return self._chunk_recursive(text)

    def _chunk_recursive(self, text):
        separators = ["\n\n", "\n", " ", ""]
        return self._recursive_split(text, separators)

    def _recursive_split(self, text, separators):
        words = text.split()
        if len(words) <= self.max_words:
            return [text]

        if not separators:
            # Fallback: split by max_words with overlap
            chunks = []
            for i in range(0, len(words), self.max_words - self.overlap_words):
                chunk_words = words[i:i + self.max_words]
                if chunk_words:
                    chunks.append(" ".join(chunk_words))
            return chunks

        separator = separators[0]
        next_separators = separators[1:]

        if separator == "":
            splits = list(text)
        else:
            splits = text.split(separator)

        chunks = []
        current_chunk = []
        current_words = 0

        for split in splits:
            if not split.strip():
                continue
            split_words = len(split.split())
            
            if current_words + split_words <= self.max_words:
                current_chunk.append(split)
                current_words += split_words
            else:
                if not current_chunk:
                    # Recursive split single huge block
                    chunks.extend(self._recursive_split(split, next_separators))
                else:
                    chunks.append(separator.join(current_chunk))
                    # Compute overlap
                    overlap_splits = []
                    overlap_words_cnt = 0
                    for s in reversed(current_chunk):
                        s_words = len(s.split())
                        if overlap_words_cnt + s_words <= self.overlap_words:
                            overlap_splits.insert(0, s)
                            overlap_words_cnt += s_words
                        else:
                            break
                    current_chunk = overlap_splits + [split]
                    current_words = sum(len(s.split()) for s in current_chunk)

        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

    def _chunk_semantically(self, text):
        """
        Groups sentences by checking similarity shifts using embeddings.
        """
        sentences = split_into_sentences(text)
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # Compute sentence embeddings
        embeddings = self.embedding_function(sentences)
        
        # Calculate cosine similarity between adjacent sentences
        similarities = []
        for idx in range(len(embeddings) - 1):
            emb1 = embeddings[idx]
            emb2 = embeddings[idx + 1]
            # Normalization
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(emb1, emb2) / (norm1 * norm2)
            else:
                sim = 0.0
            similarities.append(sim)

        chunks = []
        current_chunk = [sentences[0]]
        current_words = len(sentences[0].split())

        for idx, sim in enumerate(similarities):
            sentence = sentences[idx + 1]
            sentence_words = len(sentence.split())

            # Split if similarity drops below threshold or max words exceeded
            if sim < self.similarity_threshold or current_words + sentence_words > self.max_words:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                # New chunk starts
                current_chunk = [sentence]
                current_words = sentence_words
            else:
                current_chunk.append(sentence)
                current_words += sentence_words

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def _chunk_sentence_sliding(self, text):
        """
        Original sentence-based sliding window.
        """
        sentences = split_into_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_chunk = []
        current_words = 0

        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_words = len(sentence.split())

            if current_words + sentence_words <= self.max_words:
                current_chunk.append(sentence)
                current_words += sentence_words
                i += 1
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    overlap = current_chunk[-self.overlap_sentences:]
                    current_chunk = overlap.copy()
                    current_words = sum(len(s.split()) for s in current_chunk)
                else:
                    chunks.append(sentence)
                    i += 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks