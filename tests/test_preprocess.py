"""
Unit tests for text preprocessing & linguistic signal extraction
Author: Khushi Yadav
"""

import unittest
from src.preprocessing import clean_text, extract_linguistic_signals, expand_contractions

class TestPreprocessing(unittest.TestCase):
    def test_expand_contractions(self):
        text = "don't won't can't it's"
        expanded = expand_contractions(text)
        self.assertNotIn("don't", expanded)
        self.assertIn("cannot", expanded)

    def test_clean_text_urls_and_phones(self):
        sample = "Check this link https://bit.ly/fake-offer now! Call +91 9876543210 immediately."
        cleaned = clean_text(sample)
        self.assertIn("url_link", cleaned)
        self.assertIn("phone_number", cleaned)

    def test_extract_linguistic_signals_urgent(self):
        urgent_text = "URGENT ALERT: SHARE THIS TO 10 GROUPS IMMEDIATELY!!!"
        signals = extract_linguistic_signals(urgent_text)
        self.assertGreater(signals["uppercase_ratio"], 0.4)
        self.assertGreaterEqual(signals["exclamation_density"], 1.0)
        self.assertGreaterEqual(signals["urgency_score"], 2)
        self.assertGreater(signals["risk_index"], 40.0)

    def test_extract_linguistic_signals_calm(self):
        calm_text = "The satellite was launched from the Sriharikota space centre on Wednesday morning."
        signals = extract_linguistic_signals(calm_text)
        self.assertLess(signals["uppercase_ratio"], 0.15)
        self.assertEqual(signals["exclamation_density"], 0.0)
        self.assertEqual(signals["urgency_score"], 0)
        self.assertEqual(signals["risk_index"], 0.0)

if __name__ == "__main__":
    unittest.main()
