from mythos.config import MythosConfig
from mythos.train import train_run


def test_smoke_train_loss_finite(tmp_path):
    config = MythosConfig.from_yaml("configs/test.yaml")
    config.name = "smoke-train"
    ckpt_dir = tmp_path / "checkpoints" / config.name
    metrics = train_run(config, steps=25, checkpoint_dir=ckpt_dir, device="cpu")
    assert metrics["steps"] >= 20
    assert metrics["val_bpb"] < float("inf")
    assert metrics["params"] > 0
