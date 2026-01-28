from typing import Dict


class GeminiClient:
    """
    Safe stub Gemini client.

    This avoids crashes when Gemini SDK is unavailable.
    You can re-enable real Gemini later without changing
    any reasoning code.
    """

    def __init__(self, *args, **kwargs):
        print("⚠️ Gemini SDK not available. Using local fallback.")

    def check_consistency(
        self,
        claim: str,
        evidence: str,
    ) -> Dict[str, str]:
        """
        Deterministic fallback behavior:
        - Never crashes
        - Always returns valid output
        """

        return {
            "decision": "CONSISTENT",
            "explanation": "Local fallback used (Gemini SDK unavailable).",
        }
