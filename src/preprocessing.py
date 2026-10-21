"""
Text Preprocessing & WhatsApp Linguistic Signal Extractor
Author: Khushi Yadav
Project: Fake News & WhatsApp Misinformation Detector
"""

import re
import string
from typing import Dict, Any, Tuple

# Common WhatsApp forward chain headers & footers
FORWARD_PATTERNS = [
    r"(?i)\*?forwarded as received\*?",
    r"(?i)\*?forwarded message\*?",
    r"(?i)forward to (?:at least )?\d+ (?:groups|people|friends|contacts)",
    r"(?i)share (?:this )?with (?:everyone|all groups|your family|all contacts)",
    r"(?i)do not delete without sharing",
    r"(?i)share widely",
    r"(?i)spread this message"
]

# Urgency and sensationalist keywords frequently seen in viral misinformation
URGENCY_KEYWORDS = {
    "urgent", "alert", "breaking", "warning", "emergency", "immediately",
    "confirmed", "secret", "leaked", "danger", "shocking", "miracle",
    "cure", "deadly", "unesco", "nasa", "demonitized", "delet", "share",
    "forward", "free", "lottery", "blocked", "kyc", "discontinued",
    "impound", "radiation", "5g", "poison", "toxin", "instant"
}

# Common English Contractions
CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am"
}

def expand_contractions(text: str) -> str:
    """Replaces common contractions with expanded form."""
    for cont, expanded in CONTRACTIONS.items():
        text = text.replace(cont, expanded)
    return text

def extract_linguistic_signals(text: str) -> Dict[str, Any]:
    """
    Extracts heuristic signals that characterize viral WhatsApp forwards:
    - High uppercase ratio (shouting)
    - Exclamation and question mark density
    - Forward chain indicators
    - Urgency triggers
    - URL/link presence
    """
    if not text:
        return {
            "char_length": 0,
            "word_count": 0,
            "uppercase_ratio": 0.0,
            "exclamation_density": 0.0,
            "question_density": 0.0,
            "forward_markers_found": 0,
            "urgency_score": 0,
            "has_url": 0,
            "risk_index": 0.0
        }

    char_length = len(text)
    words = text.split()
    word_count = len(words)

    # Uppercase ratio (excluding spaces and punctuation)
    alpha_chars = [c for c in text if c.isalpha()]
    upper_chars = [c for c in alpha_chars if c.isupper()]
    uppercase_ratio = len(upper_chars) / len(alpha_chars) if alpha_chars else 0.0

    # Punctuation counts
    exclamations = text.count('!')
    questions = text.count('?')
    exclamation_density = (exclamations / char_length) * 100 if char_length > 0 else 0.0
    question_density = (questions / char_length) * 100 if char_length > 0 else 0.0

    # Forward markers
    forward_markers_found = sum(1 for pattern in FORWARD_PATTERNS if re.search(pattern, text))

    # Urgency keyword count
    lower_text = text.lower()
    cleaned_tokens = re.findall(r'\b[a-z0-9]+\b', lower_text)
    urgency_score = sum(1 for token in cleaned_tokens if token in URGENCY_KEYWORDS)

    # URL presence
    has_url = 1 if re.search(r'https?://\S+|www\.\S+|bit\.ly/\S+', text) else 0

    # Combined heuristic risk score (0 to 100)
    risk_points = 0
    if uppercase_ratio > 0.15:
        risk_points += min(30, int(uppercase_ratio * 100))
    if exclamations >= 2:
        risk_points += min(25, exclamations * 5)
    if forward_markers_found > 0:
        risk_points += 25
    if urgency_score > 0:
        risk_points += min(20, urgency_score * 7)
    if has_url:
        risk_points += 15

    risk_index = min(100.0, float(risk_points))

    return {
        "char_length": char_length,
        "word_count": word_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "exclamation_density": round(exclamation_density, 3),
        "question_density": round(question_density, 3),
        "forward_markers_found": forward_markers_found,
        "urgency_score": urgency_score,
        "has_url": has_url,
        "risk_index": round(risk_index, 1)
    }

def clean_text(text: str, remove_forward_tags: bool = False) -> str:
    """
    Standard NLP text cleaning pipeline:
    - Normalizes unicode whitespace
    - Handles contractions
    - Strips noisy URLs while keeping domain context if relevant
    - Cleans redundant punctuation but retains semantic structure
    """
    if not isinstance(text, str):
        return ""

    # Replace newlines and excessive spaces
    text = re.sub(r'[\r\n\t]+', ' ', text)

    # Expand contractions
    text = expand_contractions(text)

    # Optional: strip WhatsApp forward tags if isolating core claim
    if remove_forward_tags:
        for pattern in FORWARD_PATTERNS:
            text = re.sub(pattern, '', text)

    # Normalize URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' url_link ', text)

    # Normalize phone numbers
    text = re.sub(r'\+?\d[\d -]{8,}\d', ' phone_number ', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text
