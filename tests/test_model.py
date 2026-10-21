"""
Unit tests for model pipeline inference and LIME explainability
Author: Khushi Yadav
"""

import unittest
from src.predict import NewsPredictor

class TestModelInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = NewsPredictor()

    def test_suspicious_prediction(self):
        hoax = "URGENT WARNING: Drinking boiled garlic with hot water kills 100% coronavirus in 1 hour! Forward to 10 groups before midnight!!!"
        res = self.predictor.predict(hoax, explain=False)
        self.assertEqual(res["prediction"], "Suspicious")
        self.assertGreater(res["confidence"], 50.0)

    def test_reliable_prediction(self):
        news = "The Indian Space Research Organisation announced that the navigation satellite successfully reached designated orbit on Wednesday morning."
        res = self.predictor.predict(news, explain=False)
        self.assertEqual(res["prediction"], "Reliable")
        self.assertGreater(res["confidence"], 50.0)

    def test_lime_explanation_structure(self):
        sample = "URGENT: Secret document leaked from ministry, share immediately!"
        res = self.predictor.predict(sample, explain=True)
        self.assertIsNotNone(res["explanation"])
        self.assertIn("html_highlighted", res["explanation"])
        self.assertIn("all_features", res["explanation"])
        self.assertGreater(len(res["explanation"]["all_features"]), 0)

if __name__ == "__main__":
    unittest.main()
