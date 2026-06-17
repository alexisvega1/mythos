from pathlib import Path

from mythos.config import MythosConfig
from mythos.eval.composite import compute_mythos_score
from mythos.eval.harness import run_full_eval
from mythos.train import train_run


def test_eval_requires_checkpoint():
    config = MythosConfig.from_yaml("configs/test.yaml")
    raw = run_full_eval(model_path=None, config=config, limit=3)
    composite = compute_mythos_score(raw)
    assert raw.val_bpb is None
    assert composite.mythos_score is None


def test_eval_after_train(tmp_path):
    config = MythosConfig.from_yaml("configs/test.yaml")
    config.name = "eval-smoke"
    ckpt_dir = tmp_path / "checkpoints" / config.name
    metrics = train_run(config, steps=30, checkpoint_dir=ckpt_dir, device="cpu")
    raw = run_full_eval(model_path=metrics["checkpoint"], config=config, limit=3)
    assert raw.val_bpb is not None
    composite = compute_mythos_score(raw)
    assert composite.mythos_score is not None
