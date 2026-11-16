from pydantic import BaseModel, Field

from foundation.rag.bm25 import BM25Index
from foundation.rag.chunking import Chunk


class HybridResult(BaseModel):
    """Résultat d'une recherche hybride combinée."""

    chunk: Chunk = Field(..., description="Le chunk extrait")
    rrf_score: float = Field(..., description="Le score fusionné RRF")
    dense_rank: int | None = Field(
        None, description="Rang dans la recherche vectorielle"
    )
    sparse_rank: int | None = Field(None, description="Rang dans la recherche BM25")


def reciprocal_rank_fusion(
    dense_chunks: list[Chunk],
    sparse_chunks: list[Chunk],
    k: int = 60,
) -> list[HybridResult]:
    """Combine deux listes ordonnées de chunks via la méthode RRF.

    Args:
        dense_chunks: Liste des chunks ordonnés par ChromaDB.
        sparse_chunks: Liste des chunks ordonnés par BM25.
        k: Constante de lissage (par défaut 60).

    Returns:
        Liste de HybridResult triée par rrf_score décroissant.
    """
    scores: dict[int, float] = {}
    dense_ranks: dict[int, int] = {}
    sparse_ranks: dict[int, int] = {}
    chunk_map: dict[int, Chunk] = {}

    for rank, chunk in enumerate(dense_chunks, start=1):
        c_id = chunk.chunk_id
        chunk_map[c_id] = chunk
        dense_ranks[c_id] = rank
        scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank))

    for rank, chunk in enumerate(sparse_chunks, start=1):
        c_id = chunk.chunk_id
        chunk_map[c_id] = chunk
        sparse_ranks[c_id] = rank
        scores[c_id] = scores.get(c_id, 0.0) + (1.0 / (k + rank))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    results: list[HybridResult] = []
    for c_id in sorted_ids:
        results.append(
            HybridResult(
                chunk=chunk_map[c_id],
                rrf_score=scores[c_id],
                dense_rank=dense_ranks.get(c_id),
                sparse_rank=sparse_ranks.get(c_id),
            )
        )

    return results


def hybrid_search(
    query: str,
    chunks: list[Chunk],
    chroma_results: list[Chunk],
    top_k: int = 3,
) -> list[HybridResult]:
    """Exécute une recherche hybride complète (BM25 + Dense + RRF)."""
    bm25_index = BM25Index(chunks)
    sparse_results = [res.chunk for res in bm25_index.search(query, top_k=10)]

    fused_results = reciprocal_rank_fusion(
        dense_chunks=chroma_results, sparse_chunks=sparse_results
    )
    return fused_results[:top_k]
