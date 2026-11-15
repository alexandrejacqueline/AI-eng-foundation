from typing import Any, cast

import chromadb

from foundation.llm.client import GeminiClient
from foundation.rag.chunking import Chunk, chunk_text


def load_and_chunk_file(file_path: str) -> list[Chunk]:
    """Lit un fichier local et le découpe en objets Chunk avec métadonnées."""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    chunks = chunk_text(content)
    return chunks


def index_chunks(chunks: list[Chunk]) -> chromadb.Collection:
    """Indexe la liste de Chunks et leurs métadonnées dans ChromaDB."""
    client = chromadb.Client()
    collection = client.get_or_create_collection("pricing")

    documents = [c.text for c in chunks]
    ids = [f"chunk_{c.chunk_id}" for c in chunks]
    metadatas = [
        {
            "chunk_id": c.chunk_id,
            "start_char": c.start_char,
            "end_char": c.end_char,
        }
        for c in chunks
    ]

    collection.add(
        documents=documents,
        ids=ids,
        metadatas=cast(Any, metadatas),
    )

    return collection


def search(
    collection: chromadb.Collection, question: str, n_results: int = 2
) -> dict[str, Any]:
    """Exécute une recherche vectorielle par similarité dans ChromaDB."""
    results = collection.query(query_texts=[question], n_results=n_results)
    return cast(dict[str, Any], results)


def ask(question: str) -> str:
    """Interroge le RAG complet en injectant les chunks pertinents au LLM."""
    chunks = load_and_chunk_file("data/sample/pricing_notes.md")
    collection = index_chunks(chunks)

    results = search(collection, question, n_results=2)
    documents = results.get("documents")
    context_chunks = documents[0] if documents else []
    context = "\n\n".join(context_chunks)

    prompt = f"""Réponds à la question en t'appuyant uniquement sur le contexte suivant.
Si l'information n'est pas dans le contexte, dis-le clairement.

Contexte :
{context}

Question : {question}
"""

    client = GeminiClient()
    response = client.chat(prompt)
    return response.answer
