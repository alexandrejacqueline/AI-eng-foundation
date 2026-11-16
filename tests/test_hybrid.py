from foundation.rag.chunking import Chunk
from foundation.rag.hybrid import reciprocal_rank_fusion


def test_reciprocal_rank_fusion():
    """Vérifie que RRF fait remonter un chunk présent dans les deux classements."""
    chunk_a = Chunk(text="Produit A", chunk_id=0, start_char=0, end_char=9)
    chunk_b = Chunk(text="Produit B", chunk_id=1, start_char=10, end_char=19)
    chunk_c = Chunk(text="Produit C", chunk_id=2, start_char=20, end_char=29)

    # Dense place B en 1er, A en 2e
    dense_chunks = [chunk_b, chunk_a]
    # BM25 place A en 1er, C en 2e
    sparse_chunks = [chunk_a, chunk_c]

    results = reciprocal_rank_fusion(dense_chunks, sparse_chunks, k=60)

    # Chunk A est 2e en dense et 1er en sparse -> son score RRF doit le placer au sommet
    assert results[0].chunk.chunk_id == 0
    assert results[0].dense_rank == 2
    assert results[0].sparse_rank == 1
