from foundation.rag.chunking import chunk_text


def test_chunk_text_splits_long_text() -> None:
    text = "a" * 50
    chunks = chunk_text(text, max_chars=20, overlap=5)
    assert len(chunks) >= 2
    assert all(len(c) <= 20 for c in chunks)


def test_chunk_text_empty() -> None:
    assert chunk_text("   ") == []
