from rag.parsing import parse_docs, DOCS_DIR
from rag.chunking import chunk_sections
from rag.retrieval.lexical import build_index, search
from rag.retrieval.filtering import filter_indices


def get_all_chunks():
    manual_chunks = chunk_sections(parse_docs(DOCS_DIR / "manual"))
    wiki_chunks = chunk_sections(parse_docs(DOCS_DIR / "wiki"))
    return manual_chunks + wiki_chunks


def test_filter_correctness():
    chunks = get_all_chunks()
    bm25 = build_index(chunks)
    query = "How to reset adapter setting?"

    indices = filter_indices(chunks, source="manual")
    filtered = search(query, chunks, bm25, len(chunks), allowed_indices=indices)

    assert len(filtered) > 0
    for chunk, _ in filtered:
        assert chunk.source == "manual"


def test_filter_recall_preservation():
    chunks = get_all_chunks()
    bm25 = build_index(chunks)
    query = "How to reset adapter setting?"

    indices = {i for i, chunk in enumerate(chunks) if chunk.file_path.name == "err_4000.md"}
    assert len(indices) == 5

    results = search(query, chunks, bm25, 5, allowed_indices=indices)
    assert len(results) == 5

    results = search(query, chunks, bm25, 3, allowed_indices=indices)
    assert len(results) == 3
