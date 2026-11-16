import math
import re
from collections import Counter

from pydantic import BaseModel, Field

from foundation.rag.chunking import Chunk


def tokenize(text: str) -> list[str]:
    """Nettoie et découpe un texte en jetons (tokens) minuscules."""
    return re.findall(r"\w+", text.lower())


class BM25Result(BaseModel):
    """Résultat d'une recherche BM25 avec son score."""

    chunk: Chunk = Field(..., description="Le chunk correspondant")
    score: float = Field(..., description="Le score BM25 du chunk")


class BM25Index:
    """Index BM25 (Okapi) en Python pur pour la recherche lexicale."""

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.corpus_size = len(chunks)

        self.tokenized_corpus = [tokenize(c.text) for c in chunks]
        self.doc_len = [len(doc) for doc in self.tokenized_corpus]
        self.avgdl = (
            sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0
        )
        self.doc_freqs: list[Counter[str]] = [
            Counter(doc) for doc in self.tokenized_corpus
        ]

        self.nd: dict[str, int] = {}
        for doc in self.tokenized_corpus:
            for word in set(doc):
                self.nd[word] = self.nd.get(word, 0) + 1

    def search(self, query: str, top_k: int = 5) -> list[BM25Result]:
        """Recherche les top_k chunks les plus pertinents pour la requête."""
        query_tokens = tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        scores = [0.0] * self.corpus_size
        for token in query_tokens:
            if token not in self.nd:
                continue
            n_q = self.nd[token]
            idf = math.log((self.corpus_size - n_q + 0.5) / (n_q + 0.5) + 1.0)
            for i, freq in enumerate(self.doc_freqs):
                if token in freq:
                    f = freq[token]
                    numerator = f * (self.k1 + 1.0)
                    denominator = f + self.k1 * (
                        1.0 - self.b + self.b * (self.doc_len[i] / self.avgdl)
                    )
                    scores[i] += idf * (numerator / denominator)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[BM25Result] = []
        for idx, score in indexed_scores[:top_k]:
            if score > 0:
                results.append(BM25Result(chunk=self.chunks[idx], score=score))

        return results
