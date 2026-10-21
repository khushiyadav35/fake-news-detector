"""
Feature Extraction Pipeline
Author: Khushi Yadav
Project: Fake News & WhatsApp Misinformation Detector
"""

from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_vectorizer(
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    sublinear_tf: bool = True
) -> TfidfVectorizer:
    """
    Constructs an optimized TF-IDF Vectorizer tuned for short-to-medium text
    and WhatsApp forward phrasing patterns.
    
    - sublinear_tf: Replaces TF with 1 + log(TF), dampening outlier token frequency
    - ngram_range: (1, 2) captures unigrams ('lemon', 'cancer') and bigrams ('hot water', 'bank account')
    - token_pattern: Retains standard words and alphanumeric patterns
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
        stop_words='english',
        strip_accents='unicode',
        norm='l2'
    )
