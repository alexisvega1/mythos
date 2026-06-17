"""Standalone visual demo server for Project Mythos.

Serves a single-page UI plus a small JSON API backed by the REAL trained
checkpoints (base + SFT) and the defensive safety router. Reuses
mythos.serve.inference and mythos.router — does not touch the Lane B serve app.

    python demo/build_demo.py      # once, to train + write assets
    python demo/serve_demo.py      # then open http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from mythos.router import route_request
from mythos.serve.inference import MythosEngine

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"

app = FastAPI(title="Mythos Demo")
_ENGINES: dict[str, MythosEngine] = {}


def _load_engines(device: str | None) -> None:
    for key, path in {"base": ASSETS / "ckpt" / "latest.pt", "sft": ASSETS / "ckpt-sft" / "latest.pt"}.items():
        if path.exists():
            try:
                _ENGINES[key] = MythosEngine(path, device=device or "cpu")
            except Exception as exc:  # noqa: BLE001
                print(f"warn: could not load {key} engine: {exc}")


class GenRequest(BaseModel):
    prompt: str
    model: str = "base"
    max_tokens: int = 48
    temperature: float = 0.7


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "index.html")


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    path = ASSETS / "run.json"
    if not path.exists():
        return JSONResponse({"error": "run demo/build_demo.py first"}, status_code=503)
    return JSONResponse(json.loads(path.read_text()))


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "engines": sorted(_ENGINES.keys())}


@app.post("/api/generate")
def generate(req: GenRequest) -> dict:
    # Defensive safety router first — flagged prompts are not run through the model.
    decision = route_request(req.prompt, access_tier="public")
    if decision.use_fallback:
        return {
            "routed": True, "reason": decision.reason, "model": decision.model_tier,
            "text": f"⚠ Routed to the Fable safety tier ({decision.reason}). "
                    f"This is a defensive demo; the model has no such capability.",
            "prompt_tokens": 0, "completion_tokens": 0,
        }
    engine = _ENGINES.get(req.model) or _ENGINES.get("base")
    if engine is None:
        return {"routed": False, "text": "[no checkpoint — run demo/build_demo.py]",
                "model": "unavailable", "prompt_tokens": 0, "completion_tokens": 0}
    text, pt, ct = engine.generate(req.prompt, max_tokens=req.max_tokens, temperature=req.temperature)
    return {"routed": False, "text": text or "(empty)", "model": f"mythos-{req.model}",
            "prompt_tokens": pt, "completion_tokens": ct}


def main() -> None:
    parser = argparse.ArgumentParser(description="Mythos visual demo server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    _load_engines(args.device)
    print(f"Engines loaded: {sorted(_ENGINES.keys()) or 'NONE (run build_demo.py)'}")
    print(f"Open http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
