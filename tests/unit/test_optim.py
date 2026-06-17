from mythos.config import MythosConfig
from mythos.optim import build_optimizer, lr_schedule


def test_lr_schedule_warmup():
    lr = lr_schedule(0, warmup=10, max_steps=100, base_lr=1e-3)
    assert lr == 0.0
    lr_mid = lr_schedule(5, warmup=10, max_steps=100, base_lr=1e-3)
    assert 0 < lr_mid < 1e-3


def test_build_optimizer_splits_params():
    from mythos.model import GPT

    config = MythosConfig(depth=2, n_head=2, n_embd=64, vocab_size=128)
    model = GPT.from_config(config)
    muon, adam = build_optimizer(model, "nor_muon", 1e-3, 0.1)
    assert len(muon.param_groups[0]["params"]) > 0
    assert len(adam.param_groups[0]["params"]) > 0
