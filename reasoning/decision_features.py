"""
Feature extraction for calibrated decision layer.

This module converts structured reasoning outputs
(from ConsistencyChecker) into a compact numeric
feature vector suitable for lightweight classifiers.

Design goals:
- Interpretable
- Robust to missing / partial evidence
- Faithful to narrative constraint reasoning
"""

from typing import List, Dict
import numpy as np


def extract_decision_features(details: List[Dict]) -> np.ndarray:
    """
    Convert reasoning details into a numeric feature vector.

    Args:
        details: List of dicts produced by ConsistencyChecker, each containing:
            - decision: "CONSISTENT" or "CONTRADICTION"
            - max_similarity: float (optional)

    Returns:
        np.ndarray of shape (5,)
        Feature order:
            0: number of constraints
            1: number of contradictions
            2: maximum evidence similarity
            3: average evidence similarity
            4: contradiction ratio
    """

    if not details or not isinstance(details, list):
        # No constraints → highly uncertain → contradiction leaning
        return np.array(
            [0.0, 0.0, 0.0, 0.0, 1.0],
            dtype=float,
        )

    n_constraints = len(details)

    n_contradictions = sum(
        1 for d in details
        if str(d.get("decision", "")).upper() == "CONTRADICTION"
    )

    similarities = []

    for d in details:
        sim = d.get("max_similarity", None)
        if isinstance(sim, (int, float)):
            similarities.append(float(sim))

    if similarities:
        max_sim = max(similarities)
        avg_sim = float(np.mean(similarities))
    else:
        max_sim = 0.0
        avg_sim = 0.0

    contradiction_ratio = (
        n_contradictions / n_constraints
        if n_constraints > 0
        else 1.0
    )
    
    features = np.array(
        [
            float(n_constraints),
            float(n_contradictions),
            float(max_sim),
            float(avg_sim),
            float(contradiction_ratio),
        ],
        dtype=float,
    )

    return features
