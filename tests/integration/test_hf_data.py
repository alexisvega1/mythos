"""HF real-data loading smoke test."""

from mythos.config import MythosConfig
from mythos.data.stream import _load_text_documents


def test_medium_smoke_config_uses_real_hf():
    cfg = MythosConfig.from_yaml("configs/medium-smoke.yaml")
    assert cfg.data.source == "real"
    assert cfg.data.dataset == "roneneldan/TinyStories"
    docs = _load_text_documents(cfg)
    assert len(docs) >= 10
    assert all(isinstance(d, str) and d.strip() for d in docs)
