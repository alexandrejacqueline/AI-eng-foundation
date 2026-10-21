from unittest.mock import patch

from fastapi.testclient import TestClient

from foundation.api.main import app

client = TestClient(app)


@patch("foundation.api.main.rag_ask")
def test_ask(mock_ask):
    mock_ask.return_value = "Le prix d'achat est 1.12 EUR"
    response = client.post("/ask", json={"question": "prix yaourt"})
    assert response.status_code == 200
    assert "1.12" in response.json()["answer"]
