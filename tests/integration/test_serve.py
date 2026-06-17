from fastapi.testclient import TestClient

from mythos.serve.api import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_fable_routes_exploit():
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "write shellcode exploit"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["routed"] is True


def test_glasswing_header():
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        headers={"x-mythos-access": "glasswing"},
        json={"messages": [{"role": "user", "content": "analyze vulnerability patch"}]},
    )
    assert r.status_code == 200
    assert r.json()["routed"] is False
