from pathlib import Path

import chromadb
import pandas as pd
from chromadb.utils import embedding_functions


def row_to_text(row: pd.Series) -> str:
    """Transforme une ligne de tableau en une description textuelle naturelle."""
    mdd_status = "Marque Distributeur (MDD)" if row["is_mdd"] else "Marque Nationale"
    return (
        f"Produit : {row['libelle_produit']}. "
        f"Marque : {row['marque']} ({mdd_status}). "
        f"Catégorie : {row['categorie']}. "
        f"Prix de vente : {row['prix_eur']} EUR."
    )


def build_pricing_vectorstore(
    csv_path: Path | str,
    db_dir: Path | str = "./data/chroma_db",
    collection_name: str = "pricing_mdd",
) -> chromadb.Collection:
    """Ingère un CSV de prix dans une collection ChromaDB persistante."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Le fichier CSV est introuvable : {csv_path}")

    df = pd.read_csv(csv_path)

    client = chromadb.PersistentClient(path=str(db_dir))
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
    )

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for _, row in df.iterrows():
        documents.append(row_to_text(row))
        metadatas.append(
            {
                "code_ean": str(row["code_ean"]),
                "libelle": str(row["libelle_produit"]),
                "marque": str(row["marque"]),
                "categorie": str(row["categorie"]),
                "prix_eur": float(row["prix_eur"]),
                "is_mdd": bool(row["is_mdd"]),
            }
        )
        ids.append(f"ean_{row['code_ean']}")

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"✅ Ingestion réussie : {len(documents)} produits ajoutés à ChromaDB.")
    return collection
