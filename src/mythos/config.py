from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataMixConfig:
    fineweb_edu: float = 0.85
    stack_v2: float = 0.10
    synthetic_code: float = 0.05
    cyber_defensive: float = 0.0
    bio_scientific: float = 0.0


@dataclass
class DataConfig:
    source: str = "real"  # real | synthetic | fixture
    dataset: str = "roneneldan/TinyStories"
    dataset_split: str = "train"
    tokenizer: str = "gpt2"
    fixture_path: str = "data/fixtures/tiny_corpus.txt"
    val_holdout_pct: int = 10
    seed: int = 42
    max_hf_documents: int = 512
    pretrain_tokens_b: float = 0.5
    mix: DataMixConfig = field(default_factory=DataMixConfig)


@dataclass
class EvalConfig:
    proxy_every: int = 50
    full_every: int = 500
    val_batches: int = 20


@dataclass
class PostTrainConfig:
    swe_rft: bool = False
    swe_grpo: bool = False
    cyber_ladder_rl: bool = False


@dataclass
class ServeConfig:
    context_length: int = 8192
    max_output: int = 4096
    fallback_model: str = "mythos-medium"


@dataclass
class MythosConfig:
    name: str = "mythos-nano"
    depth: int = 12
    n_head: int = 6
    n_embd: int = 768
    block_size: int = 1024
    vocab_size: int = 50257
    batch_size: int = 32
    grad_accum: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    max_steps: int = 10000
    train_budget_seconds: int = 300
    optimizer: str = "nor_muon"
    use_rope: bool = True
    use_qk_norm: bool = True
    activation: str = "relu2"
    untied_embeddings: bool = True
    use_sliding_window: bool = False
    sliding_window: int = 4096
    val_bpb_gate: float = 1.01
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    posttrain: PostTrainConfig = field(default_factory=PostTrainConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> MythosConfig:
        raw = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> MythosConfig:
        data_raw = raw.get("data", {})
        mix_raw = data_raw.get("mix", {})
        eval_raw = raw.get("eval", {})
        post_raw = raw.get("posttrain", {})
        serve_raw = raw.get("serve", {})

        return cls(
            name=raw.get("name", "mythos-nano"),
            depth=raw.get("depth", 12),
            n_head=raw.get("n_head", 6),
            n_embd=raw.get("n_embd", 768),
            block_size=raw.get("block_size", 1024),
            vocab_size=raw.get("vocab_size", 50257),
            batch_size=raw.get("batch_size", 32),
            grad_accum=raw.get("grad_accum", 4),
            learning_rate=raw.get("learning_rate", 3e-4),
            weight_decay=raw.get("weight_decay", 0.1),
            warmup_steps=raw.get("warmup_steps", 100),
            max_steps=raw.get("max_steps", 10000),
            train_budget_seconds=raw.get("train_budget_seconds", 300),
            optimizer=raw.get("optimizer", "nor_muon"),
            use_rope=raw.get("use_rope", True),
            use_qk_norm=raw.get("use_qk_norm", True),
            activation=raw.get("activation", "relu2"),
            untied_embeddings=raw.get("untied_embeddings", True),
            use_sliding_window=raw.get("use_sliding_window", False),
            sliding_window=raw.get("sliding_window", 4096),
            val_bpb_gate=raw.get("val_bpb_gate", 1.01),
            data=DataConfig(
                source=data_raw.get("source", "real"),
                dataset=data_raw.get("dataset", "roneneldan/TinyStories"),
                dataset_split=data_raw.get("dataset_split", "train"),
                tokenizer=data_raw.get("tokenizer", "gpt2"),
                fixture_path=data_raw.get("fixture_path", "data/fixtures/tiny_corpus.txt"),
                val_holdout_pct=data_raw.get("val_holdout_pct", 10),
                seed=data_raw.get("seed", 42),
                max_hf_documents=data_raw.get("max_hf_documents", 512),
                pretrain_tokens_b=data_raw.get("pretrain_tokens_b", 0.5),
                mix=DataMixConfig(**{k: mix_raw.get(k, 0.0) for k in DataMixConfig.__dataclass_fields__}),
            ),
            eval=EvalConfig(
                **{k: eval_raw.get(k, getattr(EvalConfig(), k)) for k in EvalConfig.__dataclass_fields__}
            ),
            posttrain=PostTrainConfig(
                **{k: post_raw.get(k, getattr(PostTrainConfig(), k)) for k in PostTrainConfig.__dataclass_fields__}
            ),
            serve=ServeConfig(
                **{k: serve_raw.get(k, getattr(ServeConfig(), k)) for k in ServeConfig.__dataclass_fields__}
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def n_layer(self) -> int:
        return self.depth

    def sync_vocab_from_tokenizer(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
