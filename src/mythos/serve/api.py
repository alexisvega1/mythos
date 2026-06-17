"""
OpenAI-compatible Mythos/Fable dual-tier API.

Routes sensitive queries to fallback model per Fable safeguards.
"""

from __future__ import annotations

import argparse
import uuid
from typing import Literal

import uvicorn
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field

from mythos.router import route_request

app = FastAPI(title="Project Mythos API", version="0.1.0")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "mythos-frontier"
    messages: list[ChatMessage]
    max_tokens: int = 4096
    temperature: float = 0.7


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    choices: list[ChatChoice]
    routed: bool = False
    route_reason: str = ""


def _access_tier(header: str | None) -> Literal["public", "glasswing", "bio_trust"]:
    if header == "glasswing":
        return "glasswing"
    if header == "bio-trust":
        return "bio_trust"
    return "public"


def _generate_stub(content: str, model: str, routed: bool, reason: str) -> str:
    prefix = f"[{model}{' fallback' if routed else ''}]"
    if routed:
        return f"{prefix} Routed ({reason}). Safe response for: {content[:200]}"
    return f"{prefix} Mythos response for: {content[:200]}"


@app.post("/v1/chat/completions")
def chat_completions(
    req: ChatRequest,
    x_mythos_access: str | None = Header(default=None),
) -> ChatResponse:
    user_text = " ".join(m.content for m in req.messages if m.role == "user")
    tier = _access_tier(x_mythos_access)
    decision = route_request(user_text, access_tier=tier)
    model_used = decision.model_tier if not decision.use_fallback else decision.model_tier
    reply = _generate_stub(user_text, model_used, decision.use_fallback, decision.reason)
    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        model=model_used,
        choices=[ChatChoice(message=ChatMessage(role="assistant", content=reply))],
        routed=decision.use_fallback,
        route_reason=decision.reason,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "tiers": ["mythos", "fable"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos serve")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--checkpoint", default="checkpoints/latest")
    args = parser.parse_args()
    print(f"Loading checkpoint metadata from {args.checkpoint} (stub inference)")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
