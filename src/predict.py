"""
Inference & Analysis Pipeline
Author: Khushi Yadav
Project: Fake News & WhatsApp Misinformation Detector
"""

import os
import joblib
from typing import Dict, Any
from src.preprocessing import clean_text, extract_linguistic_signals
from src.explainability import NewsExplainer

class NewsPredictor:
    """
    Unified predictor combining text preprocessing, ML model inference,
    linguistic risk scoring, and LIME explainability.
    """
    def __init__(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "model_pipeline.pkl")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Run 'python -m src.train' first.")
            
        self.pipeline = joblib.load(model_path)
        self.explainer = NewsExplainer(class_names=["Reliable", "Suspicious"])

    def predict(self, raw_text: str, explain: bool = True) -> Dict[str, Any]:
        """
        Runs comprehensive analysis on the input text.
        """
        # Linguistic heuristic signals
        linguistic_signals = extract_linguistic_signals(raw_text)
        
        # Clean text for model
        cleaned = clean_text(raw_text)
        
        # Model prediction
        probas = self.pipeline.predict_proba([cleaned])[0]
        prob_reliable = float(probas[0])
        prob_suspicious = float(probas[1])
        
        # Final classification
        predicted_label = "Suspicious" if prob_suspicious >= 0.5 else "Reliable"
        confidence = prob_suspicious if predicted_label == "Suspicious" else prob_reliable
        
        result = {
            "text": raw_text,
            "cleaned_text": cleaned,
            "prediction": predicted_label,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "Reliable": round(prob_reliable, 4),
                "Suspicious": round(prob_suspicious, 4)
            },
            "linguistic_signals": linguistic_signals,
            "explanation": None
        }
        
        # LIME explainability
        if explain:
            # We wrap predict_proba to accept a list of texts and apply clean_text
            def predict_fn(texts):
                cleaned_batch = [clean_text(t) for t in texts]
                return self.pipeline.predict_proba(cleaned_batch)
                
            explanation = self.explainer.explain(raw_text, predict_fn, num_features=8)
            result["explanation"] = explanation
            
        return result
