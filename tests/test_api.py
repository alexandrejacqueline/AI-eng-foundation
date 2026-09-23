from fastapi.testclient import TestClient

from foundation.api.main import app

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ai-eng-foundation"


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
