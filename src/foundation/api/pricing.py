from typing import Any, cast

import chromadb
from chromadb.utils import embedding_functions
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/pricing", tags=["Pricing RAG"])


# --- Schémas Pydantic ---
class PricingQueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="Terme de recherche sémantique (ex: 'biscuit chocolat')",
        min_length=2,
    )
    is_mdd: bool | None = Field(
        default=None,
        description=("Filtre optionnel : True pour MDD, False pour Marque Nationale"),
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Nombre de résultats à retourner",
    )


class ProductResult(BaseModel):
    code_ean: str
    libelle: str
    marque: str
    categorie: str
    prix_eur: float
    is_mdd: bool
    text_description: str


class PricingQueryResponse(BaseModel):
    query: str
    results_count: int
    products: list[ProductResult]


# --- Route Endpoint ---
@router.post("/ask", response_model=PricingQueryResponse)
def ask_pricing(payload: PricingQueryRequest) -> PricingQueryResponse:
    """Interroge la base vectorielle ChromaDB sur le catalogue de prix."""
    try:
        client = chromadb.PersistentClient(path="./data/chroma_db")
        embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        collection = client.get_collection(
            name="pricing_mdd",
            embedding_function=cast(Any, embedding_fn),
        )
    except Exception as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"La collection ChromaDB 'pricing_mdd' est inaccessible : {err}",
        ) from err

    # Filtre métadonnées optionnel
    where_clause: dict[str, Any] = {}
    if payload.is_mdd is not None:
        where_clause = {"is_mdd": payload.is_mdd}

    # Recherche vectorielle
    results = collection.query(
        query_texts=[payload.query],
        n_results=payload.top_k,
        where=cast(Any, where_clause) if where_clause else None,
    )

    products: list[ProductResult] = []
    if results and results["documents"] and results["metadatas"]:
        for doc, meta in zip(
            results["documents"][0], results["metadatas"][0], strict=False
        ):
            products.append(
                ProductResult(
                    code_ean=str(meta["code_ean"]),
                    libelle=str(meta["libelle"]),
                    marque=str(meta["marque"]),
                    categorie=str(meta["categorie"]),
                    prix_eur=float(str(meta["prix_eur"])),
                    is_mdd=bool(meta["is_mdd"]),
                    text_description=doc,
                )
            )

    return PricingQueryResponse(
        query=payload.query,
        results_count=len(products),
        products=products,
    )
