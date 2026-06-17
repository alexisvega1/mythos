from __future__ import annotations

from pathlib import Path

import torch

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

from mythos.checkpoint import load_checkpoint


@register_model("mythos")
class MythosLMEval(LM):
    """Minimal lm-eval wrapper around a Mythos GPT checkpoint."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", batch_size: int = 1, **kwargs):
        super().__init__()
        self.device = device
        self._batch_size = batch_size
        self.model, self.config, _ = load_checkpoint(checkpoint, device=device)
        self.model.eval()
        self._rank = 0
        self._world_size = 1

    @classmethod
    def create_from_arg_string(cls, arg_string: str, additional_config=None):
        args = dict(item.split("=", 1) for item in arg_string.split(",") if "=" in item)
        checkpoint = args.get("checkpoint", args.get("pretrained", ""))
        device = args.get("device", "cpu")
        return cls(checkpoint=checkpoint, device=device)

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self):
        return self.config.block_size

    @property
    def max_gen_toks(self):
        return 64

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = torch.device(value)

    def tok_encode(self, string: str, **kwargs):
        from mythos.data.stream import get_tokenizer

        enc = get_tokenizer(self.config.data.tokenizer)
        return enc.encode(string)

    def tok_decode(self, tokens):
        from mythos.data.stream import get_tokenizer

        enc = get_tokenizer(self.config.data.tokenizer)
        return enc.decode(tokens)

    def _model_call(self, inps):
        with torch.no_grad():
            logits, _ = self.model(inps)
        return logits

    def loglikelihood(self, requests):
        from mythos.data.stream import get_tokenizer

        enc = get_tokenizer(self.config.data.tokenizer)
        results = []
        for context, continuation in requests:
            ctx = enc.encode(context)
            cont = enc.encode(continuation)
            inp = torch.tensor([ctx + cont[:-1]], dtype=torch.long, device=self.device)
            tgt = torch.tensor([ctx + cont], dtype=torch.long, device=self.device)
            logits = self._model_call(inp)
            log_probs = torch.log_softmax(logits, dim=-1)
            seq_logprob = 0.0
            cont_start = len(ctx) - 1
            for i, token_id in enumerate(cont):
                pos = cont_start + i
                if pos >= log_probs.size(1):
                    break
                seq_logprob += float(log_probs[0, pos, token_id].item())
            greedy = int(logits[0, cont_start : cont_start + len(cont)].argmax(dim=-1).eq(torch.tensor(cont, device=self.device)).all())
            results.append((seq_logprob, bool(greedy)))
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def generate_until(self, requests):
        raise NotImplementedError
