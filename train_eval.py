
import os
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from pathway_pipeline.indexing import build_embedding_index
from reasoning.constraint_extractor import ConstraintExtractor
from reasoning.consistency_checker import ConsistencyChecker
from reasoning.rationale_builder import RationaleBuilder
from reasoning.decision_features import extract_decision_features
from reasoning.calibrated_decider import CalibratedDecider
from sentence_transformers import SentenceTransformer


TRAIN_CSV = "data/train.csv"
REPORT_DIR = "evaluation/reports"
EMBED_MODEL = "all-MiniLM-L6-v2"

LABEL_MAP = {
    "consistent": 1,
    "contradict": 0,
}


def evaluate_on_train():
    os.makedirs(REPORT_DIR, exist_ok=True)

    # ---------- LOAD DATA ----------
    df = pd.read_csv(TRAIN_CSV)
    print("📊 TRAIN DATA SHAPE:", df.shape)
    print(df.head())

    if len(df) < 2:
        raise ValueError(
        f"Train dataset too small: {len(df)} rows found. "
        "Check data/train.csv"
    )

    required_cols = {"id", "content", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Train CSV missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    print("🔹 Building narrative embedding index (once)...")
    embedding_table = build_embedding_index()


    claim_embedder = SentenceTransformer(EMBED_MODEL)
    extractor = ConstraintExtractor()
    checker = ConsistencyChecker(embedding_table)
    rationale_builder = RationaleBuilder()


    feature_matrix = []
    labels = []
    cached_samples = []
    
    print("🔹 Running reasoning pipeline (feature collection)...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        sample_id = row["id"]
        backstory = row["content"]
        
        raw_label = str(row["label"]).strip().lower()
        true_label = LABEL_MAP[raw_label]
        
        constraints = extractor.extract_constraints(backstory)
        
        if not constraints:
            outcome = {
                "details": [{
                    "claim": "<no constraints>",
                    "decision": "CONTRADICTION",
                    "max_similarity": 0.0,
                    "explanation": "No constraints extracted"
                }]
            }
        else:
            claim_texts = [c["claim"] for c in constraints]
            claim_embeddings = claim_embedder.encode(
                claim_texts,
                normalize_embeddings=True,
            )
            outcome = checker.check_constraints(
                constraints=constraints,
                claim_embeddings=claim_embeddings,
            )

        features = extract_decision_features(outcome["details"])

        feature_matrix.append(features)
        labels.append(true_label)
        cached_samples.append((sample_id, backstory, outcome, true_label))




    X = np.vstack(feature_matrix)
    y = np.array(labels)

    print("\n📊 FEATURE STATISTICS")
    print("Samples:", X.shape[0])
    print("Unique labels:", np.unique(y, return_counts=True))
    
    if len(np.unique(y)) > 1:
        decider = CalibratedDecider()
        decider.train(X, y)
        use_decider = True
        print("✅ Calibration model trained")
    else:
        decider = None
        use_decider = False
        print("⚠️ Calibration skipped (only one class present)")


    y_true = []
    y_pred = []
    error_rows = []
    
    print("🔹 Evaluating with calibrated decision layer...")
    
    for sample_id, backstory, outcome, true_label in cached_samples:
        features = extract_decision_features(outcome["details"])


        if features[1] == 0:
            pred_label = 1
        else:
            pred_label = 0
            
        y_true.append(true_label)
        y_pred.append(pred_label)

        if pred_label != true_label:
            rationale = rationale_builder.build(
                label=pred_label,
                details=outcome["details"],
            )
            error_rows.append(
                {
                    "id": sample_id,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "backstory": backstory,
                    "rationale": rationale,
                }
            )
        if use_decider:
            pred_label = decider.predict(features)
        else:

            pred_label = 1 if features[1] == 0 else 0


    acc = accuracy_score(y_true, y_pred)

    print("\n🔥 CALIBRATED TRAIN ACCURACY:", round(acc * 100, 2), "%\n")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
            digits=3,
        )
    )

    print("📊 Prediction distribution:", Counter(y_pred))
    print("📊 Ground truth distribution:", Counter(y_true))

 
    if error_rows:
        error_df = pd.DataFrame(error_rows)
        error_path = os.path.join(REPORT_DIR, "train_errors.csv")
        error_df.to_csv(error_path, index=False)
        print(f"❌ Errors saved to {error_path}")
    else:
        print("🎯 No errors found — perfect training accuracy!")


    print("\n📌 Feature importance (calibration layer):")
    for k, v in decider.get_feature_importance().items():
        print(f"  {k}: {round(v, 4)}")

    return acc


if __name__ == "__main__":
    evaluate_on_train()
