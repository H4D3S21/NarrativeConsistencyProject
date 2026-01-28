import os
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from pathway_pipeline.indexing import build_embedding_index
from reasoning.constraint_extractor import ConstraintExtractor
from reasoning.consistency_checker import ConsistencyChecker
from reasoning.rationale_builder import RationaleBuilder


TEST_CSV = "data/test.csv"
OUTPUT_DIR = "results"
OUTPUT_FILE = "results.csv"
EMBED_MODEL = "all-MiniLM-L6-v2"


def run_prediction():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(TEST_CSV)

    required_cols = {"id", "backstory"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Test CSV must contain columns: {required_cols}"
        )

    print("🔹 Building narrative embedding index (once)...")
    embedding_table = build_embedding_index()

    claim_embedder = SentenceTransformer(EMBED_MODEL)
    extractor = ConstraintExtractor()
    checker = ConsistencyChecker(embedding_table)
    rationale_builder = RationaleBuilder()

    results = []

    print("🔹 Running inference on test data...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["id"]
        backstory = row["backstory"]

        constraints = extractor.extract_constraints(backstory)

        claim_texts = [c["claim"] for c in constraints]
        claim_embeddings = claim_embedder.encode(
            claim_texts,
            normalize_embeddings=True,
        )

        outcome = checker.check_constraints(
            constraints=constraints,
            claim_embeddings=claim_embeddings,
        )

        label = outcome["label"]

        rationale = rationale_builder.build(
            label=label,
            details=outcome["details"],
        )

        results.append(
            {
                "id": sample_id,
                "label": label,
                "rationale": rationale,
            }
        )

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    pd.DataFrame(results).to_csv(output_path, index=False)

    print(f"✅ Prediction complete. Results saved to {output_path}")


if __name__ == "__main__":
    run_prediction()
