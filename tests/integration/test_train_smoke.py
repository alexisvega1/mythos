from mythos.config import MythosConfig
from mythos.train import train_run


def test_smoke_train_loss_finite():
    config = MythosConfig(
        depth=2,
        n_head=2,
        n_embd=64,
        vocab_size=128,
        block_size=16,
        batch_size=2,
        grad_accum=1,
        max_steps=20,
        train_budget_seconds=30,
    )
    metrics = train_run(config, steps=15, budget_seconds=30)
    assert metrics["steps"] >= 10
    assert metrics["val_bpb"] < float("inf")
    assert metrics["params"] > 0
