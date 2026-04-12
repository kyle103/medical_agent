import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_register_ok():
    client = TestClient(app)
    r = client.post("/api/v1/user/register", json={"user_nickname": "a"})
    assert r.status_code == 200
    assert r.json()["data"]["user_id"]
