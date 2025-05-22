from fastapi.testclient import TestClient
from api import app


client=TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the ML Model API"}



def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    json_resp = response.json()
    assert "status" in json_resp
    assert json_resp["status"] in ["healthy", "model not loaded"]



