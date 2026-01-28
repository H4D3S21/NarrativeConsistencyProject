import os
import pathway as pw


DATA_DIR = "data/novels"

class NovelSchema(pw.Schema):
    doc_id: str
    title: str
    text: str


def ingest_novels():
    """
    Ingest novels and return a Pathway table.
    """

    rows = []

    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Novel directory not found: {DATA_DIR}")

    for filename in sorted(os.listdir(DATA_DIR)):
        if not filename.lower().endswith(".txt"):
            continue

        file_path = os.path.join(DATA_DIR, filename)

        doc_id = filename.replace(".txt", "").strip().lower()
        title = (
            filename.replace(".txt", "")
            .replace("_", " ")
            .replace("-", " ")
            .title()
        )

        with open(file_path, "r", encoding="utf-8-sig") as f:
            text = f.read().strip()

        if not text:
            continue

        rows.append((doc_id, title, text))

    if not rows:
        raise ValueError(f"No valid novels found in {DATA_DIR}")

    novels_table = pw.debug.table_from_rows(
        rows=rows,
        schema=NovelSchema,
    )

    return novels_table
