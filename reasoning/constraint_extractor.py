import re
from typing import List, Dict


class ConstraintExtractor:
    """
    Extracts structured constraints from a hypothetical backstory.
    This is deterministic and rule-based (no LLM).
    """

    def __init__(self):
        self.claim_patterns = [
            r"\b(always|never|could not|couldn't|refused to)\b",
            r"\b(from childhood|since childhood|as a child)\b",
            r"\b(believed that|was convinced that|feared that)\b",
            r"\b(vowed to|swore to|promised to)\b",
            r"\b(hated|feared|trusted|despised)\b",
            r"\b(determined to|resolved to)\b",
        ]

    def split_sentences(self, text: str) -> List[str]:
        """
        Basic sentence splitter.
        """
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def is_strong_claim(self, sentence: str) -> bool:
        """
        Check whether a sentence expresses a strong narrative constraint.
        """
        sentence_lower = sentence.lower()
        for pattern in self.claim_patterns:
            if re.search(pattern, sentence_lower):
                return True
        return False

    def classify_claim(self, sentence: str) -> str:
        """
        Classify constraint type based on sentence content.
        """

        s = sentence.lower()

        if any(k in s for k in ["childhood", "as a child", "early life"]):
            return "early_life_constraint"

        if any(k in s for k in ["feared", "afraid", "terrified"]):
            return "psychological_fear"

        if any(k in s for k in ["believed", "convinced", "assumed"]):
            return "belief_constraint"

        if any(k in s for k in ["vowed", "swore", "promised", "resolved"]):
            return "commitment_constraint"

        if any(k in s for k in ["hated", "despised", "resented"]):
            return "emotional_constraint"

        return "general_constraint"

    def extract_constraints(self, backstory: str) -> List[Dict]:
        """
        Main extraction function.

        Returns a list of constraints:
        [
          {
            "claim": str,
            "type": str
          }
        ]
        """

        constraints = []
        sentences = self.split_sentences(backstory)

        for sentence in sentences:
            if self.is_strong_claim(sentence):
                constraint = {
                    "claim": sentence,
                    "type": self.classify_claim(sentence),
                }
                constraints.append(constraint)


        if not constraints and backstory.strip():
            constraints.append(
                {
                    "claim": backstory.strip(),
                    "type": "soft_constraint",
                }
            )

        return constraints


if __name__ == "__main__":
    sample = """
    From childhood, the character feared authority figures.
    He believed that trust always leads to betrayal.
    Later in life, he vowed to never work with the state.
    """

    extractor = ConstraintExtractor()
    constraints = extractor.extract_constraints(sample)

    for c in constraints:
        print(c)
