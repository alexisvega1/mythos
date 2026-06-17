from __future__ import annotations

from pathlib import Path

import pytest

from mythos.checkpoint import save_checkpoint
from mythos.config import MythosConfig
from mythos.eval.composite import compute_mythos_score
from mythos.eval.harness import run_full_eval
from mythos.model import GPT
from mythos.train import train_run


@pytest.fixture
def test_config() -> MythosConfig:
    return MythosConfig.from_yaml("configs/test.yaml")


def test_no_checkpoint_means_unavailable(test_config):
    raw = run_full_eval(model_path="/nonexistent/checkpoint.pt", config=test_config)
    assert raw.val_bpb is None
    composite = compute_mythos_score(raw)
    assert composite.mythos_score is None


def test_random_init_vs_trained_bpb(test_config, tmp_path):
    test_config.name = "test-no-fake"
    ckpt_dir = tmp_path / "checkpoints" / test_config.name
    ckpt_dir.mkdir(parents=True)

    model = GPT.from_config(test_config)
    random_ckpt = ckpt_dir / "random.pt"
    save_checkpoint(random_ckpt, model, test_config, step=0)

    random_raw = run_full_eval(model_path=str(random_ckpt), config=test_config, limit=3)
    assert random_raw.val_bpb is not None

    metrics = train_run(test_config, steps=80, checkpoint_dir=ckpt_dir)
    trained_raw = run_full_eval(model_path=metrics["checkpoint"], config=test_config, limit=3)
    assert trained_raw.val_bpb is not None
    assert trained_raw.val_bpb < random_raw.val_bpb


def test_different_checkpoints_differ(test_config, tmp_path):
    test_config.name = "test-diff"
    ckpt_dir = tmp_path / "checkpoints" / test_config.name
    ckpt_dir.mkdir(parents=True)

    m1 = train_run(test_config, steps=40, checkpoint_dir=ckpt_dir, device="cpu")
    r1 = run_full_eval(model_path=m1["checkpoint"], config=test_config, limit=3)

    m2 = train_run(test_config, steps=120, checkpoint_dir=ckpt_dir, device="cpu")
    r2 = run_full_eval(model_path=m2["checkpoint"], config=test_config, limit=3)

    assert r1.val_bpb != r2.val_bpb


def test_no_limit_derived_constants(test_config, tmp_path):
    test_config.name = "test-limit"
    ckpt_dir = tmp_path / "checkpoints" / test_config.name
    metrics = train_run(test_config, steps=50, checkpoint_dir=ckpt_dir, device="cpu")
    for limit in [3, 7, 11]:
        raw = run_full_eval(model_path=metrics["checkpoint"], config=test_config, limit=limit)
        assert raw.val_bpb is not None
        assert raw.val_bpb != pytest.approx(0.02 * limit, abs=0.05)
        assert raw.gsm8k_acc != pytest.approx(min(0.15, 0.02 * limit), abs=0.01)
