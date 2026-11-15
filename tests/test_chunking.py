import pytest

from foundation.rag.chunking import Chunk, chunk_text


def test_chunk_text_empty():
    """Un texte vide doit retourner une liste vide."""
    assert chunk_text("") == []


def test_chunk_text_short():
    """Un texte plus court que chunk_size donne un seul chunk."""
    text = "Petit texte."
    result = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(result) == 1
    assert result[0] == Chunk(text=text, chunk_id=0, start_char=0, end_char=len(text))


def test_chunk_text_sliding_window_offsets():
    """Vérifie le chevauchement exact et la reconstruction via les offsets."""
    sample_text = "0123456789"
    # chunk_size=5, chunk_overlap=2 => step=3
    # Chunk 0: "01234" (0..5)
    # Chunk 1: "34567" (3..8) -> overlap "34"
    # Chunk 2: "6789"  (6..10) -> overlap "67"
    chunks = chunk_text(sample_text, chunk_size=5, chunk_overlap=2)

    assert len(chunks) == 3

    assert chunks[0].text == "01234"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == 5

    assert chunks[1].text == "34567"
    assert chunks[1].start_char == 3
    assert chunks[1].end_char == 8

    assert chunks[2].text == "6789"
    assert chunks[2].start_char == 6
    assert chunks[2].end_char == 10

    # Vérification universelle : le découpage par index correspond au texte du chunk
    for c in chunks:
        assert sample_text[c.start_char : c.end_char] == c.text


def test_chunk_text_invalid_params():
    """Vérifie que les entrées invalides lèvent bien des ValueError."""
    with pytest.raises(ValueError, match="chunk_size doit être strictement positif"):
        chunk_text("test", chunk_size=0, chunk_overlap=0)

    with pytest.raises(
        ValueError, match="chunk_overlap doit être strictement inférieur"
    ):
        chunk_text("test", chunk_size=100, chunk_overlap=100)
