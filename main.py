import os
import pandas as pd
from tqdm import tqdm

from pathway_pipeline.indexing import build_embedding_index
from reasoning.constraint_extractor import ConstraintExtractor
from reasoning.consistency_checker import ConsistencyChecker
from sentence_transformers import SentenceTransformer


TEST_CSV = "data/test.csv"
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "results.csv")
EMBED_MODEL = "all-MiniLM-L6-v2"
# --------------------------------------

def build_real_rationale(outcome: dict) -> str:
    """
    Return ONLY the real evidence scenario (claim)
    that caused the prediction.
    No templates. No AI-style text.
    """

    details = outcome.get("details", [])
    if not details:
        return "No explicit narrative constraint identified"

    top = max(
        details,
        key=lambda d: float(d.get("max_similarity", 0.0))
    )

    claim = top.get("claim", "").strip()

    if not claim:
        return "Narrative evidence unclear"

    return claim[:120]



def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(TEST_CSV)

    if not {"id", "content"}.issubset(df.columns):
        raise ValueError("test.csv must contain columns: id, content")

    print("🔹 Building narrative embedding index (Pathway)...")
    embedding_table = build_embedding_index()

    claim_embedder = SentenceTransformer(EMBED_MODEL)
    extractor = ConstraintExtractor()
    checker = ConsistencyChecker(embedding_table)

    results = []

    print("🔹 Running inference...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["id"]
        text = row["content"]

        constraints = extractor.extract_constraints(text)

        if not constraints:
            results.append({
                "id": sample_id,
                "prediction": 1,
                "rationale": "No explicit constraints extracted from backstory",
            })
            continue

        claims = [c["claim"] for c in constraints]
        embeddings = claim_embedder.encode(claims, normalize_embeddings=True)

        outcome = checker.check_constraints(
            constraints=constraints,
            claim_embeddings=embeddings,
        )

        prediction = int(outcome["label"])
        rationale = build_real_rationale(outcome)

        results.append({
            "id": sample_id,
            "prediction": prediction,
            "rationale": rationale,
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ results.csv written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
