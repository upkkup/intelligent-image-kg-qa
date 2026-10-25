from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_short_question_rejected():
    response = client.post("/api/qa", json={"question": "x"})
    assert response.status_code == 422
