"""Text embedding generation using TF-IDF.

Embeddings convert text into numerical vectors that capture semantic meaning.
These vectors are used for similarity search — finding chunks that are
semantically related to a user's query.

Why TF-IDF for this project:
- Zero external dependencies (no model downloads)
- Fast and deterministic
- Good baseline before moving to neural embeddings
- Understandable — the math is transparent

In production, you'd use sentence-transformers or OpenAI embeddings
for better semantic understanding.
"""
import logging
from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class TfidfEmbedder:
    """Generate TF-IDF embeddings for text chunks.

    TF-IDF (Term Frequency-Inverse Document Frequency) weights words by:
    - TF: How often a word appears in THIS document
    - IDF: How rare a word is ACROSS ALL documents
    Result: Common words (the, is) get low scores; distinctive words get high scores.
    """

    def __init__(self, max_features: int = 5000) -> None:
        self.vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._is_fitted = False
        self._embeddings: np.ndarray = np.array([])

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit vectorizer on corpus and transform texts to embeddings."""
        self._embeddings = self.vectorizer.fit_transform(texts).toarray()
        self._is_fitted = True
        logger.info("Generated embeddings: shape=%s, vocab_size=%d", self._embeddings.shape, len(self.vectorizer.vocabulary_))
        return self._embeddings

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform new texts using already-fitted vectorizer."""
        if not self._is_fitted:
            raise RuntimeError("Must call fit_transform() before transform()")
        return self.vectorizer.transform(texts).toarray()

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
