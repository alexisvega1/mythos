"""One-command Mythos demo — install, build (if needed), serve, open browser."""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "demo"
ASSETS = DEMO / "assets"
CKPT_BASE = ASSETS / "ckpt" / "latest.pt"
CKPT_SFT = ASSETS / "ckpt-sft" / "latest.pt"
RUN_JSON = ASSETS / "run.json"
VENV = ROOT / ".venv"
VENV_PY = VENV / "bin" / "python"


def _free_port(start: int = 8000) -> int:
    for port in range(start, start + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    return start


def _ensure_venv(install: bool) -> Path:
    if not VENV_PY.exists():
        print("→ Creating virtualenv .venv …")
        subprocess.check_call([sys.executable, "-m", "venv", str(VENV)], cwd=ROOT)
    if install:
        print("→ Installing mythos (core deps only — skip lm-eval) …")
        subprocess.check_call(
            [str(VENV_PY), "-m", "pip", "install", "-e", str(ROOT), "-q"],
            cwd=ROOT,
        )
    return VENV_PY


def _assets_ready() -> bool:
    return CKPT_BASE.exists() and CKPT_SFT.exists() and RUN_JSON.exists()


def _build(py: Path, *, quick: bool, force: bool) -> None:
    if _assets_ready() and not force:
        print("→ Checkpoints ready — skipping train (use --rebuild to retrain)")
        return
    cmd = [str(py), str(DEMO / "build_demo.py")]
    if quick:
        cmd.append("--quick")
    print("→ Training demo model on Shakespeare + SFT (~1 min on Apple Silicon) …")
    subprocess.check_call(cmd, cwd=ROOT)


def _device_flag(py: Path) -> str:
    out = subprocess.check_output(
        [str(py), "-c", "import torch; print('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))"],
        cwd=ROOT,
        text=True,
    ).strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mythos live demo — train, serve, open browser (one command)",
    )
    parser.add_argument("--no-install", action="store_true", help="Skip pip install")
    parser.add_argument("--rebuild", action="store_true", help="Force retrain even if checkpoints exist")
    parser.add_argument("--quick", action="store_true", help="Faster train (80 steps) for CPU demos")
    parser.add_argument("--port", type=int, default=0, help="Port (0 = auto)")
    parser.add_argument("--no-open", action="store_true", help="Do not open browser")
    args = parser.parse_args()

    os.chdir(ROOT)
    py = _ensure_venv(install=not args.no_install)
    _build(py, quick=args.quick, force=args.rebuild)

    port = args.port or _free_port(8000)
    device = _device_flag(py)
    url = f"http://127.0.0.1:{port}"

    print()
    print("=" * 56)
    print("  MYTHOS LIVE — frontier-style lab, honest metrics")
    print(f"  {url}")
    print(f"  device={device}  engines=base+sft  streaming=on")
    print("=" * 56)
    print("  Press Ctrl+C to stop.")
    print()

    if not args.no_open:
        time.sleep(0.8)
        webbrowser.open(url)

    subprocess.call(
        [str(py), str(DEMO / "serve_demo.py"), "--host", "127.0.0.1", "--port", str(port), "--device", device],
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
