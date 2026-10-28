import math
import re
import numpy as np


class BM25Okapi:
    """
    A lightweight, dependency-free implementation of the BM25Okapi algorithm.
    Fits dynamically on raw texts on startup.
    """

    def __init__(self, corpus_texts, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_texts)
        
        # Tokenize corpus
        tokenized_corpus = [self._tokenize(text) for text in corpus_texts]
        
        # Document lengths
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        
        # Document frequencies and term frequencies per doc
        self.doc_freqs = []
        self.idf = {}
        
        self._initialize(tokenized_corpus)

    def _tokenize(self, text):
        """
        Simple, robust regex tokenizer that lowercases and extracts alphanumeric words.
        """
        if not text:
            return []
        return re.findall(r"\w+", text.lower())

    def _initialize(self, tokenized_corpus):
        # Count document frequency of each word
        word_doc_counts = {}
        for doc in tokenized_corpus:
            frequencies = {}
            for word in doc:
                frequencies[word] = frequencies.get(word, 0) + 1
            self.doc_freqs.append(frequencies)
            
            # Count document frequency
            for word in set(doc):
                word_doc_counts[word] = word_doc_counts.get(word, 0) + 1
                
        # Calculate IDF for each word
        for word, count in word_doc_counts.items():
            # BM25 standard IDF with positive correction (adding 1.0 inside log)
            self.idf[word] = math.log((self.corpus_size - count + 0.5) / (count + 0.5) + 1.0)

    def get_scores(self, query):
        """
        Calculates BM25 scores for all documents given a query string.
        """
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size)
        
        for word in query_tokens:
            if word not in self.idf:
                continue
            
            idf_val = self.idf[word]
            
            for idx in range(self.corpus_size):
                freq = self.doc_freqs[idx].get(word, 0)
                if freq == 0:
                    continue
                
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[idx] / self.avg_doc_len
                )
                
                scores[idx] += idf_val * (numerator / denominator)
                
        return scores
