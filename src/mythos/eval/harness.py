from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

from mythos.checkpoint import load_checkpoint
from mythos.config import MythosConfig
from mythos.data.stream import byte_weighted_bpb, get_batch_iterator
from mythos.eval.composite import RawScores

logger = logging.getLogger(__name__)


def evaluate_held_out_bpb(
    checkpoint_path: str | Path,
    device: str | None = None,
    max_batches: int = 20,
) -> float:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_checkpoint(checkpoint_path, device=dev)
    batches = get_batch_iterator(config, split="val", batch_size=min(config.batch_size, 4))
    return byte_weighted_bpb(
        model,
        batches,
        config.data.tokenizer,
        dev,
        max_batches=max_batches,
    )


def run_lm_eval_tasks(
    checkpoint_path: str | Path,
    tasks: list[str] | None = None,
    limit: int | None = None,
) -> dict[str, float | None]:
    """Run lm-eval on the Mythos checkpoint when installed and supported; else unavailable.

    Per PLAN.md, a metric we cannot compute is reported as *unavailable* (None) — never a
    crash and never a fake constant. A task is unavailable if lm-eval is absent OR if it
    requires a capability the wrapper does not implement (e.g. gsm8k needs ``generate_until``,
    which is out of scale for a tiny from-scratch model — see OLMES guidance in PLAN.md).
    """
    tasks = tasks or ["gsm8k"]
    try:
        from mythos.eval.lm_wrapper import MythosLMEval
        from lm_eval import simple_evaluate
    except ImportError:
        return {f"{t}_acc": None for t in tasks}

    try:
        model = MythosLMEval(checkpoint_path)
        results = simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=0,
            limit=limit,
            batch_size=1,
        )
    except Exception as exc:
        # Covers NotImplementedError (e.g. gsm8k -> generate_until) and any runtime/download
        # failure: mark the tasks unavailable rather than crashing the whole eval.
        logger.warning("lm-eval tasks %s unavailable (%s): %s", tasks, type(exc).__name__, exc)
        return {f"{t}_acc": None for t in tasks}

    out: dict[str, float | None] = {}
    for task in tasks:
        task_res = results.get("results", {}).get(task, {})
        metric = task_res.get("acc,none") or task_res.get("acc_norm,none")
        out[f"{task}_acc"] = float(metric) if metric is not None else None
    return out


def run_sec_comprehension(
    checkpoint_path: str | Path,
    limit: int = 10,
    device: str | None = None,
) -> dict[str, float | None]:
    from mythos.agents import cyber as sec_agent

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, config, _ = load_checkpoint(checkpoint_path, device=dev)
    acc = sec_agent.evaluate_sec_comprehension(model, config, limit=limit, device=dev)
    if acc is None:
        return {"sec_comprehension_acc": None}
    return {"sec_comprehension_acc": acc}


def run_swe_eval(limit: int = 10) -> dict[str, float | None]:
    """SWE-bench requires mini-swe-agent integration — report unavailable until wired."""
    return {"swe_bench_verified_pass_at_1": None}


def run_full_eval(
    model_path: str | None = None,
    config: MythosConfig | None = None,
    limit: int | None = None,
    mode: str = "proxy",
    device: str | None = None,
) -> RawScores:
    unavailable: list[str] = []
    merged: dict[str, Any] = {}

    if not model_path or not Path(model_path).exists():
        unavailable.extend(["val_bpb", "gsm8k", "sec_comprehension", "mmlu_science", "swe_bench_verified"])
        return RawScores(unavailable=sorted(set(unavailable)))

    eval_batches = config.eval.val_batches if config else 20
    if mode == "proxy":
        eval_batches = min(eval_batches, max(5, (limit or 5)))

    try:
        merged["val_bpb"] = evaluate_held_out_bpb(
            model_path,
            device=device,
            max_batches=eval_batches,
        )
    except Exception:
        merged["val_bpb"] = None
        unavailable.append("val_bpb")

    lm = run_lm_eval_tasks(model_path, tasks=["gsm8k"], limit=limit if mode == "proxy" else None)
    merged["gsm8k_acc"] = lm.get("gsm8k_acc")
    if merged["gsm8k_acc"] is None:
        unavailable.append("gsm8k")

    sec = run_sec_comprehension(model_path, limit=limit or 10, device=device)
    merged["sec_comprehension_acc"] = sec.get("sec_comprehension_acc")
    if merged["sec_comprehension_acc"] is None:
        unavailable.append("sec_comprehension")

    merged["mmlu_science_acc"] = None
    unavailable.append("mmlu_science")

    swe = run_swe_eval(limit=limit or 10)
    merged["swe_bench_verified_pass_at_1"] = swe.get("swe_bench_verified_pass_at_1")
    if merged["swe_bench_verified_pass_at_1"] is None:
        unavailable.append("swe_bench_verified")

    raw = RawScores(
        val_bpb=merged.get("val_bpb"),
        gsm8k_acc=merged.get("gsm8k_acc"),
        sec_comprehension_acc=merged.get("sec_comprehension_acc"),
        mmlu_science_acc=merged.get("mmlu_science_acc"),
        swe_bench_verified_pass_at_1=merged.get("swe_bench_verified_pass_at_1"),
        unavailable=sorted(set(unavailable)),
    )
    return raw


def save_results(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
