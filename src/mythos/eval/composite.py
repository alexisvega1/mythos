from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkWeights:
    swe_bench_verified: float = 0.25
    humaneval: float = 0.20
    exploitbench: float = 0.15
    gsm8k: float = 0.15
    mmlu: float = 0.10
    frontiercode: float = 0.10
    val_bpb: float = 0.05


@dataclass
class RawScores:
    swe_bench_verified_pass_at_1: float = 0.0
    humaneval_pass_at_1: float = 0.0
    exploitbench_weighted_flags: float = 0.0
    gsm8k_acc: float = 0.0
    mmlu_macro: float = 0.0
    frontiercode_proxy: float = 0.0
    val_bpb: float = 1.0


# Reference ceilings for normalization (staged targets from plan)
NORMALIZE_CEILINGS = {
    "swe_bench_verified_pass_at_1": 0.80,
    "humaneval_pass_at_1": 0.90,
    "exploitbench_weighted_flags": 1.0,
    "gsm8k_acc": 0.95,
    "mmlu_macro": 0.90,
    "frontiercode_proxy": 0.30,
    "val_bpb": 1.5,
}


def normalize(key: str, value: float) -> float:
    if key == "val_bpb":
        ceiling = NORMALIZE_CEILINGS["val_bpb"]
        return max(0.0, min(1.0, (ceiling - value) / ceiling))
    ceiling = NORMALIZE_CEILINGS.get(key, 1.0)
    return max(0.0, min(1.0, value / ceiling))


@dataclass
class CompositeResult:
    mythos_score: float
    components: dict[str, float] = field(default_factory=dict)
    raw: RawScores = field(default_factory=RawScores)


def compute_mythos_score(
    raw: RawScores,
    weights: BenchmarkWeights | None = None,
) -> CompositeResult:
    w = weights or BenchmarkWeights()
    components = {
        "swe_bench_verified": normalize("swe_bench_verified_pass_at_1", raw.swe_bench_verified_pass_at_1),
        "humaneval": normalize("humaneval_pass_at_1", raw.humaneval_pass_at_1),
        "exploitbench": normalize("exploitbench_weighted_flags", raw.exploitbench_weighted_flags),
        "gsm8k": normalize("gsm8k_acc", raw.gsm8k_acc),
        "mmlu": normalize("mmlu_macro", raw.mmlu_macro),
        "frontiercode": normalize("frontiercode_proxy", raw.frontiercode_proxy),
        "val_bpb": normalize("val_bpb", raw.val_bpb),
    }
    score = (
        w.swe_bench_verified * components["swe_bench_verified"]
        + w.humaneval * components["humaneval"]
        + w.exploitbench * components["exploitbench"]
        + w.gsm8k * components["gsm8k"]
        + w.mmlu * components["mmlu"]
        + w.frontiercode * components["frontiercode"]
        + w.val_bpb * components["val_bpb"]
    )
    return CompositeResult(mythos_score=score, components=components, raw=raw)


def raw_from_dict(d: dict[str, Any]) -> RawScores:
    return RawScores(
        swe_bench_verified_pass_at_1=float(d.get("swe_bench_verified_pass_at_1", 0.0)),
        humaneval_pass_at_1=float(d.get("humaneval_pass_at_1", 0.0)),
        exploitbench_weighted_flags=float(d.get("exploitbench_weighted_flags", 0.0)),
        gsm8k_acc=float(d.get("gsm8k_acc", 0.0)),
        mmlu_macro=float(d.get("mmlu_macro", 0.0)),
        frontiercode_proxy=float(d.get("frontiercode_proxy", 0.0)),
        val_bpb=float(d.get("val_bpb", 1.0)),
    )
