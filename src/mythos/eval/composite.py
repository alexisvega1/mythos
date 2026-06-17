from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchmarkWeights:
    val_bpb: float = 0.55
    gsm8k: float = 0.15
    sec_comprehension: float = 0.15
    mmlu_science: float = 0.10
    swe_bench_verified: float = 0.05


@dataclass
class RawScores:
    val_bpb: float | None = None
    gsm8k_acc: float | None = None
    sec_comprehension_acc: float | None = None
    mmlu_science_acc: float | None = None
    swe_bench_verified_pass_at_1: float | None = None
    unavailable: list[str] = field(default_factory=list)


NORMALIZE_CEILINGS = {
    "val_bpb": 8.0,
    "gsm8k_acc": 0.95,
    "sec_comprehension_acc": 1.0,
    "mmlu_science_acc": 0.90,
    "swe_bench_verified_pass_at_1": 0.80,
}


def normalize(key: str, value: float) -> float:
    if key == "val_bpb":
        ceiling = NORMALIZE_CEILINGS["val_bpb"]
        return max(0.0, min(1.0, (ceiling - value) / ceiling))
    ceiling = NORMALIZE_CEILINGS.get(key, 1.0)
    return max(0.0, min(1.0, value / ceiling))


@dataclass
class CompositeResult:
    mythos_score: float | None
    components: dict[str, float] = field(default_factory=dict)
    raw: RawScores = field(default_factory=RawScores)
    unavailable: list[str] = field(default_factory=list)


METRIC_FIELDS = {
    "val_bpb": ("val_bpb", "val_bpb"),
    "gsm8k_acc": ("gsm8k", "gsm8k_acc"),
    "sec_comprehension_acc": ("sec_comprehension", "sec_comprehension_acc"),
    "mmlu_science_acc": ("mmlu_science", "mmlu_science_acc"),
    "swe_bench_verified_pass_at_1": ("swe_bench_verified", "swe_bench_verified_pass_at_1"),
}


def compute_mythos_score(
    raw: RawScores,
    weights: BenchmarkWeights | None = None,
) -> CompositeResult:
    w = weights or BenchmarkWeights()
    weight_map = {
        "val_bpb": w.val_bpb,
        "gsm8k_acc": w.gsm8k,
        "sec_comprehension_acc": w.sec_comprehension,
        "mmlu_science_acc": w.mmlu_science,
        "swe_bench_verified_pass_at_1": w.swe_bench_verified,
    }

    components: dict[str, float] = {}
    weighted_sum = 0.0
    weight_total = 0.0
    unavailable = list(raw.unavailable)

    for field_name, (component_key, weight_key) in METRIC_FIELDS.items():
        value = getattr(raw, field_name)
        wt = weight_map[weight_key]
        if value is None or math.isnan(value):
            unavailable.append(component_key)
            continue
        norm = normalize(field_name, value)
        components[component_key] = norm
        weighted_sum += wt * norm
        weight_total += wt

    if weight_total <= 0:
        return CompositeResult(mythos_score=None, components=components, raw=raw, unavailable=unavailable)

    score = weighted_sum / weight_total
    return CompositeResult(mythos_score=score, components=components, raw=raw, unavailable=unavailable)


def raw_from_dict(d: dict[str, Any]) -> RawScores:
    def _opt(key: str) -> float | None:
        if key not in d or d[key] is None:
            return None
        return float(d[key])

    return RawScores(
        val_bpb=_opt("val_bpb"),
        gsm8k_acc=_opt("gsm8k_acc"),
        sec_comprehension_acc=_opt("sec_comprehension_acc"),
        mmlu_science_acc=_opt("mmlu_science_acc"),
        swe_bench_verified_pass_at_1=_opt("swe_bench_verified_pass_at_1"),
        unavailable=list(d.get("unavailable", [])),
    )
