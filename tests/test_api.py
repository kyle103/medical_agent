import pytest
import uuid
from fastapi.testclient import TestClient

from app.main import app


def test_register_ok():
    client = TestClient(app)
    phone = f"test_user_{uuid.uuid4().hex[:10]}"
    r = client.post(
        "/api/v1/user/register",
        json={"user_nickname": "a", "phone": phone, "password": "test123456"},
    )
    assert r.status_code == 200
    assert r.json()["data"]["user_id"]
