from mythos.agents.swe import evaluate_swe_proxy, run_swe_grpo, run_swe_rft
from mythos.config import MythosConfig
from pathlib import Path


def test_swe_unavailable_until_wired():
    scores = evaluate_swe_proxy(limit=5)
    assert scores["swe_bench_verified_pass_at_1"] is None


def test_swe_posttrain_stubs():
    config = MythosConfig()
    ckpt = Path("checkpoints/test/latest.pt")
    rft = run_swe_rft(config, ckpt)
    grpo = run_swe_grpo(config, ckpt)
    assert rft["status"] == "unavailable"
    assert grpo["status"] == "unavailable"
