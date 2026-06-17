from pathlib import Path

from mythos.agents.swe import MICRO_TASKS, noop_solver, oracle_solver
from mythos.checkpoint import save_checkpoint
from mythos.config import MythosConfig
from mythos.model import GPT
from mythos.posttrain import run_rft


def _base_checkpoint(tmp_path):
    cfg = MythosConfig.from_yaml("configs/test.yaml")
    cfg.name = "rft-test"
    cfg.block_size = 96  # room for code instruction + fix
    base = tmp_path / "base" / "latest.pt"
    save_checkpoint(base, GPT.from_config(cfg), cfg, step=0)
    return cfg, base


def test_rft_collects_oracle_verified_samples_and_finetunes(tmp_path):
    """A capable solver's fixes pass the execution oracle and are fine-tuned on."""
    cfg, base = _base_checkpoint(tmp_path)
    res = run_rft(
        cfg, base, solver=oracle_solver, samples_per_task=2,
        steps=30, batch_size=4, device="cpu", out_dir=tmp_path / "rft",
    )
    assert res["accepted"] == len(MICRO_TASKS)  # every fix passes its unit test
    assert res["fine_tuned"] is True
    assert res["trained_samples"] > 0
    assert res["loss_end"] < res["loss_start"]
    assert Path(res["checkpoint"]).exists()


def test_rft_zero_yield_is_reported_honestly(tmp_path):
    """Unverified candidates (buggy code fails the oracle) → no training, honest status."""
    cfg, base = _base_checkpoint(tmp_path)
    res = run_rft(
        cfg, base, solver=noop_solver, samples_per_task=2,
        device="cpu", out_dir=tmp_path / "rft0",
    )
    assert res["accepted"] == 0
    assert res["fine_tuned"] is False
    assert res["status"] == "no_accepted_samples"
