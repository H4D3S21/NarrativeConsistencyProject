from typing import Dict, List


class RationaleBuilder:
    """
    Builds a comprehensive, submission-ready rationale
    from per-constraint consistency checks.
    """

    def __init__(self):
        pass

    def build(
        self,
        label: int,
        details: List[Dict],
    ) -> str:
        """
        Construct a final rationale string.

        Args:
            label: 1 (Consistent) or 0 (Contradiction)
            details: output from ConsistencyChecker

        Returns:
            rationale: str
        """

        rationale_parts = []

        if label == 1:
            rationale_parts.append(
                "The proposed backstory is consistent with the narrative."
            )
        else:
            rationale_parts.append(
                "The proposed backstory contradicts the narrative."
            )

        for idx, item in enumerate(details, start=1):
            claim = item["claim"]
            decision = item["decision"]
            explanation = item["explanation"]

            if decision == "CONSISTENT":
                rationale_parts.append(
                    f"Claim {idx} is consistent with the story. "
                    f"{explanation}"
                )
            else:
                rationale_parts.append(
                    f"Claim {idx} is contradicted by the narrative. "
                    f"{explanation}"
                )

        return " ".join(rationale_parts)


if __name__ == "__main__":
    sample_output = {
        "label": 0,
        "details": [
            {
                "claim": "From childhood, the character feared authority.",
                "decision": "CONTRADICTION",
                "explanation": (
                    "Later chapters show voluntary cooperation with "
                    "authority figures without hesitation."
                ),
            },
            {
                "claim": "He vowed to never work with the state.",
                "decision": "CONSISTENT",
                "explanation": "No evidence contradicts this commitment.",
            },
        ],
    }

    builder = RationaleBuilder()
    rationale = builder.build(
        label=sample_output["label"],
        details=sample_output["details"],
    )

    print(rationale)
