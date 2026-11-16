from pydantic import BaseModel, Field

from foundation.rag.chunking import Chunk


class RerankedResult(BaseModel):
    """Résultat réordonné par le Reranker."""

    chunk: Chunk = Field(..., description="Le chunk évalué")
    score: float = Field(..., description="Score de pertinence du Cross-Encoder")


class Reranker:
    """Modèle de Reranking pour réordonner les candidats retenus."""

    def __init__(
        self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ) -> None:
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception:
                self._model = None
        return self._model

    def rerank(
        self, query: str, chunks: list[Chunk], top_k: int = 3
    ) -> list[RerankedResult]:
        """Réordonne les chunks candidats selon leur pertinence avec la requête.

        Args:
            query: La question de l'utilisateur.
            chunks: La liste des chunks candidats à réévaluer.
            top_k: Le nombre de résultats finaux à conserver.

        Returns:
            Liste de RerankedResult triée par score décroissant.
        """
        if not chunks:
            return []

        model = self._get_model()
        if model is not None:
            pairs = [[query, c.text] for c in chunks]
            raw_scores = model.predict(pairs)
            results = [
                RerankedResult(chunk=c, score=float(s))
                for c, s in zip(chunks, raw_scores, strict=True)
            ]
        else:
            # Mode fallback déterministe pour les tests isolés
            query_words = set(query.lower().split())
            results = [
                RerankedResult(
                    chunk=c,
                    score=float(sum(1 for w in query_words if w in c.text.lower())),
                )
                for c in chunks
            ]

        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
