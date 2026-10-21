import chromadb

from foundation.llm.client import GeminiClient
from foundation.rag.chunking import chunk_text


def load_and_chunk_file(file_path: str) -> list[str]:
    # 1. Ouvre le fichier en lecture
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    # 2. Découpe avec la fonction qu'on a déjà
    chunks = chunk_text(content)

    # 3. Retourne la liste
    return chunks


def index_chunks(chunks: list[str]):
    client = chromadb.Client()
    collection = client.get_or_create_collection("pricing")
    collection.add(documents=chunks, ids=[f"chunk_{i}" for i in range(len(chunks))])

    return collection


def search(collection, question: str, n_results: int = 2):
    results = collection.query(query_texts=[question], n_results=n_results)
    return results


def ask(question: str) -> str:
    chunks = load_and_chunk_file("data/sample/pricing_notes.md")
    collection = index_chunks(chunks)

    results = search(collection, question, n_results=2)
    context_chunks = results["documents"][0]
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
