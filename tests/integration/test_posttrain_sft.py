from pathlib import Path

from mythos.checkpoint import save_checkpoint
from mythos.config import MythosConfig
from mythos.model import GPT
from mythos.posttrain import build_sft_dataset, load_sft_examples, run_sft
from mythos.data.stream import get_tokenizer


def test_sft_dataset_masks_prompt():
    enc = get_tokenizer("gpt2")
    examples = load_sft_examples()[:5]
    X, Y = build_sft_dataset(examples, enc, block_size=64)
    assert X.shape == Y.shape
    assert X.shape[0] == 5
    # Every example must have at least one supervised (non-masked) response token,
    # and at least one masked prompt token.
    assert (Y != -100).any()
    assert (Y == -100).any()


def test_sft_decreases_loss(tmp_path):
    config = MythosConfig.from_yaml("configs/test.yaml")
    config.name = "sft-test"
    config.block_size = 64  # room for the instruction template
    # SFT from a random-init checkpoint (isolates the SFT stage; no pretrain needed).
    base = tmp_path / "base" / "latest.pt"
    save_checkpoint(base, GPT.from_config(config), config, step=0)

    res = run_sft(
        config, base, steps=60, lr=1e-3, batch_size=8,
        device="cpu", out_dir=tmp_path / "sft",
    )
    assert res["status"] == "trained"
    assert res["examples"] > 0
    # Real learning signal: SFT loss must fall over the run.
    assert res["sft_loss_end"] < res["sft_loss_start"]
    assert Path(res["checkpoint"]).exists()
