from pathlib import Path
from typing import Any, cast

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions


def row_to_text(row: pd.Series) -> str:
    mdd_status = "Marque Distributeur (MDD)" if row["is_mdd"] else "Marque Nationale"
    return (
        f"Produit : {row['libelle_produit']}. "
        f"Marque : {row['marque']} ({mdd_status}). "
        f"Catégorie : {row['categorie']}. "
        f"Prix de vente : {row['prix_eur']} EUR."
    )


def build_pricing_vectorstore(
    csv_path: str | Path,
    db_dir: str | Path = "./data/chroma_db",
    collection_name: str = "pricing_mdd",
) -> chromadb.Collection:
    """Charge un CSV de prix et indexe les produits dans ChromaDB."""
    df = pd.read_csv(csv_path)

    client = chromadb.PersistentClient(path=str(db_dir))
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()

    # cast(Any, ...) pour satisfaire mypy face aux types internes de ChromaDB
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=cast(Any, embedding_fn),
    )

    documents = [row_to_text(row) for _, row in df.iterrows()]
    ids = [str(row["code_ean"]) for _, row in df.iterrows()]
    metadatas = [
        {
            "code_ean": str(row["code_ean"]),
            "libelle": str(row["libelle_produit"]),
            "marque": str(row["marque"]),
            "categorie": str(row["categorie"]),
            "prix_eur": float(row["prix_eur"]),
            "is_mdd": bool(row["is_mdd"]),
        }
        for _, row in df.iterrows()
    ]

    collection.upsert(
        documents=documents,
        metadatas=cast(Any, metadatas),
        ids=ids,
    )

    return collection
