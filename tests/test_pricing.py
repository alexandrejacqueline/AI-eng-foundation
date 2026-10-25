from unittest.mock import patch

from fastapi.testclient import TestClient

from foundation.api.main import app

client = TestClient(app)


@patch("foundation.api.pricing.chromadb.PersistentClient")
def test_ask_pricing_endpoint(mock_chroma):
    mock_collection = mock_chroma.return_value.get_collection.return_value
    mock_collection.query.return_value = {
        "documents": [["Produit : Biscuits Nappés Chocolat. Prix : 1.25 EUR."]],
        "metadatas": [
            [
                {
                    "code_ean": "3250390001234",
                    "libelle": "Biscuits Nappés Chocolat",
                    "marque": "U",
                    "categorie": "Épicerie Sucrée",
                    "prix_eur": 1.25,
                    "is_mdd": True,
                }
            ]
        ],
    }

    response = client.post(
        "/pricing/ask",
        json={"query": "biscuit chocolat", "is_mdd": True, "top_k": 1},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["results_count"] == 1
    assert data["products"][0]["marque"] == "U"
    assert data["products"][0]["prix_eur"] == 1.25
