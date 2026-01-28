import pathway as pw
from sentence_transformers import SentenceTransformer

from pathway_pipeline.chunking import chunk_novels


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def build_embedding_index():
    """
    Build embedding index and RETURN MATERIALIZED PYTHON DATA.

    Output:
        List[dict] with keys:
            - doc_id: str
            - text: str
            - embedding: List[float]
    """

    chunks = chunk_novels()
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    table = chunks.select(
        doc_id=chunks.doc_id,
        text=chunks.text,
        embedding=pw.apply(
            lambda t: model.encode(
                t,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist(),
            chunks.text,
        ),
    )

    df = pw.debug.table_to_pandas(table)
    records = df.to_dict(orient="records")

    return records

    # Safety check
    cleaned = []
    for r in records:
        cleaned.append(
            {
                "doc_id": r["doc_id"],
                "text": r["text"],
                "embedding": r["embedding"],
            }
        )

    return cleaned


if __name__ == "__main__":
    index = build_embedding_index()
    print("Sample record:")
    print(index[0])
