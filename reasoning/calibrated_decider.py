"""
Calibrated decision layer for narrative consistency classification.

This module learns how to aggregate structured reasoning signals
into a final consistency judgment.

It intentionally avoids deep models and operates only on
interpretable reasoning features.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


class CalibratedDecider:
    """
    Lightweight classifier that maps reasoning features
    to a final consistency label.

    Label convention:
        1 = CONSISTENT
        0 = CONTRADICTION
    """

    def __init__(self):
        self.model = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            random_state=42,
        )
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the calibration model.

        Args:
            X: np.ndarray of shape (n_samples, n_features)
            y: np.ndarray of shape (n_samples,)
        """

        if X is None or y is None:
            raise ValueError("Training data cannot be None")

        if len(X) == 0:
            raise ValueError("Empty feature matrix")

        if X.shape[0] != len(y):
            raise ValueError(
                f"Feature-label size mismatch: "
                f"{X.shape[0]} vs {len(y)}"
            )

        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, features: np.ndarray) -> int:
        """
        Predict final label from a single feature vector.

        Args:
            features: np.ndarray of shape (n_features,)

        Returns:
            int: 1 (consistent) or 0 (contradict)
        """

        if not self._is_trained:
            raise RuntimeError(
                "CalibratedDecider must be trained before prediction"
            )

        if features.ndim != 1:
            raise ValueError(
                f"Expected 1D feature vector, got shape {features.shape}"
            )

        prediction = self.model.predict(features.reshape(1, -1))[0]
        return int(prediction)

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Predict probability of CONSISTENT (class 1).

        Args:
            features: np.ndarray of shape (n_features,)

        Returns:
            float: probability in [0, 1]
        """

        if not self._is_trained:
            raise RuntimeError(
                "CalibratedDecider must be trained before prediction"
            )

        if features.ndim != 1:
            raise ValueError(
                f"Expected 1D feature vector, got shape {features.shape}"
            )

        proba = self.model.predict_proba(features.reshape(1, -1))[0][1]
        return float(proba)

    def get_feature_importance(self):
        """
        Return feature weights for interpretability.

        Returns:
            dict: mapping feature index -> coefficient
        """

        if not self._is_trained:
            raise RuntimeError(
                "CalibratedDecider must be trained before inspection"
            )

        coefficients = self.model.coef_[0]

        return {
            "n_constraints": float(coefficients[0]),
            "n_contradictions": float(coefficients[1]),
            "max_similarity": float(coefficients[2]),
            "avg_similarity": float(coefficients[3]),
            "contradiction_ratio": float(coefficients[4]),
        }
