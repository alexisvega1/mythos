"""Tests for mythos-demo launcher helpers."""

from __future__ import annotations

import socket
from pathlib import Path

from mythos.demo_cli import ASSETS, CKPT_BASE, CKPT_SFT, RUN_JSON, ROOT, _assets_ready, _free_port


def test_repo_paths_resolve():
    assert ROOT.name == "mythos"
    assert (ROOT / "demo" / "serve_demo.py").exists()
    assert ASSETS == ROOT / "demo" / "assets"


def test_assets_ready_when_present():
    if CKPT_BASE.exists() and CKPT_SFT.exists() and RUN_JSON.exists():
        assert _assets_ready()


def test_free_port_returns_bindable():
    port = _free_port(18000)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))
