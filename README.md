# Kharagpur Data Science Hackathon 2026 – Narrative Consistency Checker

## Project Overview

This project addresses the **Narrative Consistency Detection** problem posed in KDSH 2026. Given a backstory or narrative passage, the system predicts whether the story is **internally consistent (1)** or **inconsistent / contradictory (0)**.

Rather than relying on end-to-end text generation or black-box classification, the solution emphasizes **evidence-driven reasoning**, extracting explicit narrative constraints and verifying them against the broader story context.

The design prioritizes **reasoning quality, robustness, and reproducibility** over raw classification accuracy, in line with the competition guidelines.

---

## Methodology

The system follows a structured reasoning pipeline:

1. **Narrative Indexing (Pathway)**
   Narrative chunks from the provided corpus are embedded and indexed using Pathway to enable efficient semantic retrieval.

2. **Constraint Extraction**
   Each backstory is analyzed to extract explicit narrative constraints (claims or expectations implied by the text).

3. **Semantic Consistency Checking**
   Extracted constraints are embedded and compared against indexed narrative context using sentence-level embeddings.

   * Strong semantic contradictions → label `0` (inconsistent)
   * No explicit contradiction → label `1` (consistent)

4. **Conservative Decision Strategy**
   The system is intentionally conservative: inconsistency is predicted **only when strong, explicit contradiction evidence is detected**. This improves reasoning reliability at the cost of lower recall for subtle contradictions.

---

## Rationale Generation

Rationales are **not AI-generated explanations**.

Each rationale is:

* Directly extracted from the **same strongest evidence** used to make the prediction
* Short (1 line), factual, and scenario-focused
* Free from templated language, summarization, or generative paraphrasing

This ensures complete alignment between:

> **Evidence → Decision → Rationale**

Example rationale output:

* `economic collapse forced migration`
* `claims loyalty but later defects to opposing side`

---

## How to Run

Ensure a clean Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Run inference on the test set:

```bash
python main.py
```

---

## Output Format

The system generates a CSV file at:

```
outputs/results.csv
```

With the following format:

```csv
id,prediction,rationale
```

* `id` → Story ID
* `prediction` → `1` (consistent) or `0` (inconsistent)
* `rationale` → Evidence-based narrative scenario (1 line)

---

## Limitations

* The system favors **precision over recall** for detecting inconsistencies.
* Subtle or implicit contradictions without strong semantic conflict may be labeled as consistent.
* No end-to-end generative modeling is used; all reasoning is extractive and similarity-based.

These trade-offs are intentional to ensure **robust, interpretable, and reproducible reasoning**, which is prioritized in the evaluation.

---

## Reproducibility

* No external APIs are required at inference time
* Deterministic pipeline with fixed models
* Results can be reproduced by running the provided code in a clean environment

---

**Author**: KDSH 2026 Participant: Darpan Khurana, Shrushti Wakchaure
**Track**: Narrative Reasoning / Consistency Detection
