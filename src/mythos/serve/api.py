"""
OpenAI-compatible Mythos/Fable dual-tier API.

- Clean queries with a loaded checkpoint → real generation from the model.
- No checkpoint → an explicit `unavailable` reply (never a fabricated response).
- Sensitive queries (cyber/bio/distillation) → routed to a safe fallback per Fable
  safeguards. The router is a defensive-deployment demo and grants no capability.
"""

from __future__ import annotations

import argparse
import os
import time
import uuid
from typing import Literal

import uvicorn
from fastapi import FastAPI, Header
from pydantic import BaseModel

from mythos.router import route_request
from mythos.serve.inference import MythosEngine, load_engine

app = FastAPI(title="Project Mythos API", version="0.2.0")

_ENGINE: MythosEngine | None = None
_ATTEMPTED = False


def set_engine(engine: MythosEngine | None) -> None:
    """Inject an engine (used by tests and explicit checkpoint loading)."""
    global _ENGINE, _ATTEMPTED
    _ENGINE, _ATTEMPTED = engine, True


def get_engine() -> MythosEngine | None:
    """Lazily load from $MYTHOS_CHECKPOINT on first use; None if unavailable."""
    global _ENGINE, _ATTEMPTED
    if not _ATTEMPTED:
        _ATTEMPTED = True
        _ENGINE = load_engine()
    return _ENGINE


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "mythos-frontier"
    messages: list[ChatMessage]
    max_tokens: int = 128
    temperature: float = 0.8


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: Usage
    # Mythos extensions (OpenAI clients ignore unknown fields):
    routed: bool = False
    route_reason: str = ""
    available: bool = True


def _prompt_from_messages(messages: list[ChatMessage]) -> str:
    """Base completion model: continue from the user text (no chat scaffolding)."""
    user = [m.content for m in messages if m.role == "user"]
    text = "\n".join(user) if user else "\n".join(m.content for m in messages)
    return text.strip()


def _access_tier(header: str | None) -> Literal["public", "glasswing", "bio_trust"]:
    if header == "glasswing":
        return "glasswing"
    if header == "bio-trust":
        return "bio_trust"
    return "public"


REFUSAL = (
    "This request was routed to the Fable safety tier ({reason}) and not served by "
    "the primary model. This is a defensive-deployment demo; the model has no such "
    "capability regardless."
)
UNAVAILABLE = (
    "[mythos] No checkpoint is loaded, so there is no real model to generate from. "
    "Set $MYTHOS_CHECKPOINT or pass --checkpoint. (No response is fabricated.)"
)


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatRequest,
    x_mythos_access: str | None = Header(default=None),
) -> ChatResponse:
    user_text = " ".join(m.content for m in req.messages if m.role == "user")
    tier = _access_tier(x_mythos_access)
    decision = route_request(user_text, access_tier=tier)

    available = True
    if decision.use_fallback:
        reply = REFUSAL.format(reason=decision.reason)
        model_used = decision.model_tier
        prompt_tokens = completion_tokens = 0
    else:
        engine = get_engine()
        if engine is None:
            reply = UNAVAILABLE
            model_used = "mythos-unavailable"
            available = False
            prompt_tokens = completion_tokens = 0
        else:
            prompt = _prompt_from_messages(req.messages)
            reply, prompt_tokens, completion_tokens = engine.generate(
                prompt, max_tokens=req.max_tokens, temperature=req.temperature
            )
            model_used = engine.config.name

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=model_used,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=reply))],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        routed=decision.use_fallback,
        route_reason=decision.reason,
        available=available,
    )


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "tiers": ["mythos", "fable"],
        "checkpoint_loaded": engine is not None,
        "checkpoint": engine.checkpoint_path if engine else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default=os.environ.get("MYTHOS_CHECKPOINT"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    if args.checkpoint:
        engine = load_engine(args.checkpoint, device=args.device)
        set_engine(engine)
        print(
            f"Loaded checkpoint {args.checkpoint}" if engine
            else f"WARNING: checkpoint {args.checkpoint} not loadable; serving 'unavailable'"
        )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
