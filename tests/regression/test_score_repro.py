from pathlib import Path

from mythos.eval.composite import compute_mythos_score
from mythos.eval.harness import run_full_eval
from mythos.train import train_run
from mythos.config import MythosConfig


def test_score_reproducible_with_same_checkpoint(tmp_path):
    config = MythosConfig.from_yaml("configs/test.yaml")
    config.name = "repro"
    ckpt_dir = tmp_path / "checkpoints" / config.name
    metrics = train_run(config, steps=40, checkpoint_dir=ckpt_dir, device="cpu")
    raw1 = run_full_eval(model_path=metrics["checkpoint"], config=config, limit=3)
    raw2 = run_full_eval(model_path=metrics["checkpoint"], config=config, limit=3)
    c1 = compute_mythos_score(raw1)
    c2 = compute_mythos_score(raw2)
    assert raw1.val_bpb == raw2.val_bpb
    assert c1.mythos_score == c2.mythos_score
