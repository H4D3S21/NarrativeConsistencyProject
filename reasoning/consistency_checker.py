from typing import List, Dict
import numpy as np

from models.gemini_client import GeminiClient


class ConsistencyChecker:
    """
    Core reasoning engine that determines whether
    a hypothetical backstory is consistent with a narrative corpus.
    """

    def __init__(
        self,
        embedding_table: List[Dict],
        similarity_threshold: float = 0.35,
        strong_similarity: float = 0.55,
        top_k: int = 5,
    ):
        """
        Args:
            embedding_table: List[dict] with keys:
                - doc_id: str
                - text: str
                - embedding: List[float]
            similarity_threshold: minimum similarity to consider evidence relevant
            strong_similarity: similarity considered strong support
            top_k: number of chunks retrieved per claim
        """
        self.embedding_table = embedding_table
        self.similarity_threshold = similarity_threshold
        self.strong_similarity = strong_similarity
        self.top_k = top_k

        try:
            self.gemini = GeminiClient()
        except Exception as e:
            print(f"⚠️ Gemini disabled: {e}")
            self.gemini = None

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity for normalized vectors."""
        return float(np.dot(a, b))

    def retrieve_evidence(
        self,
        claim_embedding: np.ndarray,
    ) -> List[Dict]:
        """
        Retrieve top-k relevant chunks based on cosine similarity.
        """

        scored_chunks = []

        for row in self.embedding_table:
            try:
                score = self.cosine_similarity(
                    claim_embedding,
                    np.asarray(row["embedding"], dtype=np.float32),
                )
            except Exception:
                continue

            if score >= self.similarity_threshold:
                scored_chunks.append(
                    {
                        "score": score,
                        "text": row.get("text", ""),
                        "doc_id": row.get("doc_id", ""),
                    }
                )

        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[: self.top_k]

    def check_constraints(
        self,
        constraints: List[Dict],
        claim_embeddings: List[np.ndarray],
    ) -> Dict:
        """
        Check all constraints against narrative evidence.

        Returns:
        {
          "label": int,
          "details": [...],
        }
        """

        assert len(constraints) == len(claim_embeddings)

        details = []

        contradiction_count = 0
        uncertain_count = 0

        for constraint, emb in zip(constraints, claim_embeddings):
            claim_text = constraint.get("claim", "")

            evidence_chunks = self.retrieve_evidence(emb)

            if not evidence_chunks:
                decision = "CONTRADICTION"
                explanation = (
                    "No supporting narrative evidence found "
                    "(treated as contradiction)."
                )
                uncertain_count += 1

                details.append(
                    {
                        "claim": claim_text,
                        "decision": decision,
                        "max_similarity": 0.0,
                        "explanation": explanation,
                    }
                )
                contradiction_count += 1
                continue

            max_score = evidence_chunks[0]["score"]
            evidence_text = "\n\n".join(
                chunk["text"] for chunk in evidence_chunks
            )

            if self.gemini is not None:
                try:
                    result = self.gemini.check_consistency(
                        claim=claim_text,
                        evidence=evidence_text,
                    )

                    decision = result.get("decision", "CONSISTENT").upper()
                    explanation = result.get(
                        "explanation",
                        "No explicit contradiction detected.",
                    )

                except Exception:

                    decision = (
                        "CONSISTENT"
                        if max_score >= self.strong_similarity
                        else "CONTRADICTION"
                    )
                    explanation = (
                        f"Gemini unavailable. "
                        f"Embedding similarity={round(max_score, 3)}"
                    )

            else:
                if max_score >= self.strong_similarity:
                    decision = "CONSISTENT"
                    explanation = (
                        f"Strong embedding support "
                        f"(similarity={round(max_score, 3)})."
                    )
                else:
                    decision = "CONTRADICTION"
                    explanation = (
                        f"Weak or ambiguous evidence "
                        f"(similarity={round(max_score, 3)})."
                    )

            if decision == "CONTRADICTION":
                contradiction_count += 1

            details.append(
                {
                    "claim": claim_text,
                    "decision": decision,
                    "max_similarity": round(max_score, 3),
                    "explanation": explanation,
                }
            )


        if contradiction_count > 0:
            final_label = 0
        else:
            final_label = 1

        return {
            "label": final_label,
            "details": details,
        }
