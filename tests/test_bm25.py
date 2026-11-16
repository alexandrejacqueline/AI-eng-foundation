from foundation.rag.bm25 import BM25Index, tokenize
from foundation.rag.chunking import Chunk


def test_tokenize():
    """Vérifie l'extraction et la mise en minuscules des jetons."""
    text = "Le Shampoing Bio (EAN 3660001) coûte 4.50 EUR!"
    tokens = tokenize(text)
    assert "shampoing" in tokens
    assert "3660001" in tokens
    assert "4" in tokens


def test_bm25_search_exact_match():
    """Vérifie que BM25 retrouve un code EAN exact en 1ère position."""
    chunks = [
        Chunk(
            text="Le Shampoing Bio MDD EAN 3660001 coûte 4.50 EUR.",
            chunk_id=0,
            start_char=0,
            end_char=50,
        ),
        Chunk(
            text="Le Savon Doux en barre coûte 2.00 EUR.",
            chunk_id=1,
            start_char=51,
            end_char=90,
        ),
        Chunk(
            text="Dentifrice fraîcheur extrême.",
            chunk_id=2,
            start_char=91,
            end_char=120,
        ),
    ]

    index = BM25Index(chunks)
    results = index.search("3660001", top_k=2)

    assert len(results) > 0
    assert results[0].chunk.chunk_id == 0
    assert results[0].score > 0


def test_bm25_empty_query():
    """Une requête vide ou sans correspondance doit renvoyer une liste vide."""
    chunks = [Chunk(text="Test de texte.", chunk_id=0, start_char=0, end_char=14)]
    index = BM25Index(chunks)
    assert index.search("") == []
    assert index.search("mot_inexistant_xyz") == []
