from foundation.rag.chunking import Chunk
from foundation.rag.reranker import Reranker


def test_reranker_sorting():
    """Vérifie le classement de l'extrait le plus pertinent."""
    chunks = [
        Chunk(
            text="Informations générales sur l'entreprise.",
            chunk_id=0,
            start_char=0,
            end_char=40,
        ),
        Chunk(
            text="Le prix du Shampoing Bio MDD EAN 3660001 est de 4.50 EUR.",
            chunk_id=1,
            start_char=41,
            end_char=98,
        ),
        Chunk(
            text="Conditions d'utilisation du site web.",
            chunk_id=2,
            start_char=99,
            end_char=136,
        ),
    ]

    reranker = Reranker()
    results = reranker.rerank(
        query="Quel est le prix du shampoing bio ?", chunks=chunks, top_k=2
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == 1
    assert results[0].score >= results[1].score
