"""
Explainable AI (XAI) Pipeline using LIME
Author: Khushi Yadav
Project: Fake News & WhatsApp Misinformation Detector
"""

import re
import html
from typing import Dict, Any, List, Tuple
import numpy as np
from lime.lime_text import LimeTextExplainer

class NewsExplainer:
    """
    Wraps LIME Text Explainer to explain model predictions
    and generate visual word attribution heatmaps.
    """
    def __init__(self, class_names: List[str] = None):
        self.class_names = class_names or ["Reliable", "Suspicious"]
        self.explainer = LimeTextExplainer(
            class_names=self.class_names,
            split_expression=r'\W+',
            bow=True,
            random_state=42
        )

    def explain(
        self,
        text: str,
        predict_proba_fn,
        num_features: int = 10,
        num_samples: int = 500
    ) -> Dict[str, Any]:
        """
        Explains the prediction of a given text instance.
        
        Returns:
            - feature_weights: List of (word, weight) tuples
            - html_highlighted: HTML string with highlighted words
            - top_suspicious_words: Words that contributed towards 'Suspicious'
            - top_reliable_words: Words that contributed towards 'Reliable'
        """
        # Generate LIME explanation for label 1 (Suspicious)
        exp = self.explainer.explain_instance(
            text,
            predict_proba_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,)
        )

        weights_list = exp.as_list(label=1)
        
        # Split into suspicious (weight > 0) and reliable (weight < 0)
        suspicious_words = [(word, weight) for word, weight in weights_list if weight > 0]
        reliable_words = [(word, abs(weight)) for word, weight in weights_list if weight < 0]

        # Generate HTML visualization
        html_highlighted = self._generate_inline_highlight(text, weights_list)

        return {
            "all_features": weights_list,
            "suspicious_features": suspicious_words,
            "reliable_features": reliable_words,
            "html_highlighted": html_highlighted,
            "intercept": float(exp.intercept[1]) if hasattr(exp, 'intercept') else 0.0
        }

    def _generate_inline_highlight(self, text: str, weights: List[Tuple[str, float]]) -> str:
        """
        Renders an HTML snippet with colored span highlights based on LIME weights:
        - Red / Coral highlight: Pushes prediction toward Suspicious
        - Green / Emerald highlight: Pushes prediction toward Reliable
        """
        weight_dict = {w.lower(): score for w, score in weights}
        max_abs = max([abs(s) for s in weight_dict.values()] + [1e-5])
        
        # Tokenize preserving delimiters
        tokens = re.split(r'(\W+)', text)
        highlighted_parts = []

        for token in tokens:
            lower_token = token.lower()
            if lower_token in weight_dict:
                score = weight_dict[lower_token]
                intensity = min(0.85, max(0.2, abs(score) / max_abs))
                
                if score > 0:
                    # Pushes toward Suspicious (Red / Coral)
                    bg_color = f"rgba(239, 68, 68, {intensity:.2f})"
                    border_color = "rgba(220, 38, 38, 0.8)"
                    tooltip = f"Pushes toward Suspicious (+{score:.3f})"
                else:
                    # Pushes toward Reliable (Green)
                    bg_color = f"rgba(34, 197, 94, {intensity:.2f})"
                    border_color = "rgba(22, 163, 74, 0.8)"
                    tooltip = f"Pushes toward Reliable (-{abs(score):.3f})"

                safe_token = html.escape(token)
                span = (
                    f'<span title="{tooltip}" style="background-color: {bg_color}; '
                    f'border-bottom: 2px solid {border_color}; padding: 2px 4px; '
                    f'border-radius: 4px; font-weight: 600; cursor: help;">'
                    f'{safe_token}</span>'
                )
                highlighted_parts.append(span)
            else:
                highlighted_parts.append(html.escape(token))

        return "".join(highlighted_parts)
