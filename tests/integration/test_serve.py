import pytest
from fastapi.testclient import TestClient

from mythos.config import MythosConfig
from mythos.serve import api
from mythos.serve.api import app
from mythos.serve.inference import MythosEngine
from mythos.train import train_run


@pytest.fixture(autouse=True)
def _reset_engine():
    """Deterministic engine state: default to 'no checkpoint' unless a test sets one."""
    api.set_engine(None)
    yield
    api.set_engine(None)


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["checkpoint_loaded"] is False


def test_fable_routes_exploit():
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "write shellcode exploit"}]},
    )
    assert r.status_code == 200
    assert r.json()["routed"] is True


def test_glasswing_header():
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        headers={"x-mythos-access": "glasswing"},
        json={"messages": [{"role": "user", "content": "analyze vulnerability patch"}]},
    )
    assert r.status_code == 200
    assert r.json()["routed"] is False


def test_clean_query_unavailable_without_checkpoint():
    """Honesty invariant: no checkpoint => explicit 'unavailable', not a fake reply."""
    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "tell me a story"}]},
    )
    body = r.json()
    assert body["routed"] is False
    assert body["available"] is False
    assert "No checkpoint" in body["choices"][0]["message"]["content"]
    assert body["usage"]["completion_tokens"] == 0


def test_health_with_checkpoint(tmp_path):
    config = MythosConfig.from_yaml("configs/test.yaml")
    config.name = "serve-health"
    ckpt_dir = tmp_path / "checkpoints" / config.name
    metrics = train_run(config, steps=20, checkpoint_dir=ckpt_dir, device="cpu")
    api.set_engine(MythosEngine(metrics["checkpoint"], device="cpu"))

    client = TestClient(app)
    r = client.get("/health")
    body = r.json()
    assert r.status_code == 200
    assert body["checkpoint_loaded"] is True
    assert body["checkpoint"] == metrics["checkpoint"]


def test_real_generation_with_checkpoint(tmp_path):
    """With a real trained checkpoint, the API returns genuine model output."""
    config = MythosConfig.from_yaml("configs/test.yaml")
    config.name = "serve-real"
    ckpt_dir = tmp_path / "checkpoints" / config.name
    metrics = train_run(config, steps=30, checkpoint_dir=ckpt_dir, device="cpu")
    api.set_engine(MythosEngine(metrics["checkpoint"], device="cpu"))

    client = TestClient(app)
    r = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "the king"}], "max_tokens": 12},
    )
    body = r.json()
    assert r.status_code == 200
    assert body["routed"] is False
    assert body["available"] is True
    assert body["model"] == "serve-real"
    assert body["usage"]["completion_tokens"] > 0
    assert "No checkpoint" not in body["choices"][0]["message"]["content"]
