import re
from typing import List
import pathway as pw

from pathway_pipeline.ingest import ingest_novels


CHUNK_SIZE_WORDS = 900
CHUNK_OVERLAP_WORDS = 150


def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def paragraph_word_count(paragraph: str) -> int:
    return len(paragraph.split())


def chunk_paragraphs(paragraphs: List[str]) -> List[str]:
    chunks = []
    current_chunk = []
    current_word_count = 0

    i = 0
    while i < len(paragraphs):
        para = paragraphs[i]
        p_words = paragraph_word_count(para)

        if p_words > CHUNK_SIZE_WORDS:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_word_count = 0

            chunks.append(para)
            i += 1
            continue

        if current_word_count + p_words <= CHUNK_SIZE_WORDS:
            current_chunk.append(para)
            current_word_count += p_words
            i += 1
        else:
            chunks.append("\n\n".join(current_chunk))

            overlap_words = 0
            overlap_chunk = []
            for p in reversed(current_chunk):
                w = paragraph_word_count(p)
                if overlap_words + w > CHUNK_OVERLAP_WORDS:
                    break
                overlap_chunk.insert(0, p)
                overlap_words += w

            current_chunk = overlap_chunk
            current_word_count = overlap_words

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def chunk_novels() -> pw.Table:
    """
    Convert novels into chunk-level Pathway table.
    """

    novels = ingest_novels()

    # 1️⃣ Create list column
    chunk_lists = novels.select(
        doc_id=novels.doc_id,
        chunk_texts=pw.apply(
            lambda text: chunk_paragraphs(
                split_into_paragraphs(text)
            ),
            novels.text,
        ),
    )

    # 2️⃣ FLATTEN — PASS COLUMN REFERENCE (NOT STRING)
    flattened = chunk_lists.flatten(chunk_lists.chunk_texts)

    # 3️⃣ Final table
    chunks = flattened.select(
        doc_id=flattened.doc_id,
        text=flattened.chunk_texts,
        word_count=pw.apply(
            lambda t: len(t.split()),
            flattened.chunk_texts,
        ),
    )

    return chunks


if __name__ == "__main__":
    table = chunk_novels()
    pw.debug.compute_and_print(table)
