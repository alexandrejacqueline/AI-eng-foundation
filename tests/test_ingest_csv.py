from pathlib import Path

import pytest

from src.foundation.ingest_csv import build_pricing_vectorstore


@pytest.fixture
def sample_csv_path(tmp_path: Path) -> Path:
    csv_file = tmp_path / "test_prices.csv"
    csv_content = (
        "code_ean,libelle_produit,marque,categorie,prix_eur,is_mdd\n"
        "1111111111111,Gâteau Chocolat 200g,U,Épicerie,1.20,True\n"
        "2222222222222,Gâteau Chocolat 200g,Lu,Épicerie,1.90,False\n"
    )
    csv_file.write_text(csv_content, encoding="utf-8")
    return csv_file


def test_ingestion_and_hybrid_query(sample_csv_path: Path, tmp_path: Path):
    db_dir = tmp_path / "chroma_test"
    collection = build_pricing_vectorstore(
        csv_path=sample_csv_path,
        db_dir=db_dir,
        collection_name="test_collection",
    )

    assert collection.count() == 2

    results = collection.query(
        query_texts=["snack sucré au chocolat"],
        n_results=1,
        where={"is_mdd": True},
    )

    matched_metadata = results["metadatas"][0][0]
    assert matched_metadata["is_mdd"] is True
    assert matched_metadata["marque"] == "U"
    assert matched_metadata["prix_eur"] == 1.20
