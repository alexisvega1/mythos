"""Standalone visual demo server for Project Mythos.

Serves a single-page UI plus JSON + SSE streaming backed by REAL checkpoints.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from mythos.router import route_request
from mythos.serve.inference import MythosEngine

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

app = FastAPI(title="Mythos Demo")
_ENGINES: dict[str, MythosEngine] = {}
_DEVICE = "cpu"


def _load_engines(device: str | None) -> None:
    global _DEVICE
    _DEVICE = device or "cpu"
    for key, path in {
        "base": ASSETS / "ckpt" / "latest.pt",
        "sft": ASSETS / "ckpt-sft" / "latest.pt",
    }.items():
        if path.exists():
            try:
                _ENGINES[key] = MythosEngine(path, device=_DEVICE)
            except Exception as exc:  # noqa: BLE001
                print(f"warn: could not load {key} engine: {exc}")


class GenRequest(BaseModel):
    prompt: str
    model: str = "base"
    max_tokens: int = 32
    temperature: float = 0.7
    top_k: int | None = 40
    top_p: float | None = 0.92
    repetition_penalty: float | None = 1.12
    stream: bool = False


SFT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"


def _prepare_prompt(req: GenRequest) -> str:
    """Wrap freeform text in the instruct template when using the SFT engine."""
    p = req.prompt.strip()
    if req.model == "sft" and "### Instruction:" not in p:
        return SFT_TEMPLATE.format(instruction=p)
    return p


def _gen_kwargs(req: GenRequest) -> dict:
    return {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_k": req.top_k,
        "top_p": req.top_p,
        "repetition_penalty": req.repetition_penalty,
    }


def _route_or_none(req: GenRequest) -> dict | None:
    decision = route_request(req.prompt, access_tier="public")
    if decision.use_fallback:
        return {
            "routed": True,
            "reason": decision.reason,
            "model": decision.model_tier,
            "text": (
                f"⚠ Routed to the Fable safety tier ({decision.reason}). "
                "This is a defensive demo; the model has no such capability."
            ),
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    return None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    path = ASSETS / "run.json"
    if not path.exists():
        return JSONResponse(
            {"error": "missing run.json — run: mythos-demo --rebuild"},
            status_code=503,
        )
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "engines": sorted(_ENGINES.keys()),
        "device": _DEVICE,
        "streaming": True,
    }


@app.post("/api/generate")
def generate(req: GenRequest):
    routed = _route_or_none(req)
    if routed:
        return routed
    engine = _ENGINES.get(req.model) or _ENGINES.get("base")
    if engine is None:
        return {
            "routed": False,
            "text": "[no checkpoint — run: mythos-demo --rebuild]",
            "model": "unavailable",
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    if req.stream:
        return _stream_response(req, engine)
    prompt = _prepare_prompt(req)
    text, pt, ct = engine.generate(prompt, **_gen_kwargs(req))
    return {
        "routed": False,
        "text": text or "(empty)",
        "model": f"mythos-{req.model}",
        "prompt_tokens": pt,
        "completion_tokens": ct,
    }


def _stream_response(req: GenRequest, engine: MythosEngine) -> StreamingResponse:
    req_id = f"demo-{uuid.uuid4().hex[:10]}"

    def events():
        text_parts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        prompt = _prepare_prompt(req)
        for piece, prompt_tokens, completion_tokens in engine.generate_stream(
            prompt, **_gen_kwargs(req),
        ):
            text_parts.append(piece)
            chunk = {
                "id": req_id,
                "delta": piece,
                "model": f"mythos-{req.model}",
                "completion_tokens": completion_tokens,
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        final = {
            "id": req_id,
            "done": True,
            "text": "".join(text_parts),
            "model": f"mythos-{req.model}",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos visual demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _load_engines(args.device)
    print(f"Engines loaded: {sorted(_ENGINES.keys()) or 'NONE (run mythos-demo --rebuild)'}")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
