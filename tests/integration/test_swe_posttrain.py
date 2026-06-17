from mythos.agents.swe import evaluate_swe_proxy, run_swe_grpo, run_swe_rft
from pathlib import Path


def test_swe_proxy():
    scores = evaluate_swe_proxy(limit=5)
    assert "swe_bench_verified_pass_at_1" in scores
    assert 0 <= scores["swe_bench_verified_pass_at_1"] <= 1


def test_swe_posttrain_stubs():
    from mythos.config import MythosConfig

    config = MythosConfig()
    ckpt = Path("checkpoints/test/latest.pt")
    rft = run_swe_rft(config, ckpt)
    grpo = run_swe_grpo(config, ckpt)
    assert rft["stage"] == "swe_rft"
    assert grpo["scaffold"] == "mini-swe-agent"
