"""Generate local eval dashboard from results JSON."""

from __future__ import annotations

import json
from pathlib import Path


def render_dashboard(results_path: Path, output_path: Path) -> None:
    data = json.loads(results_path.read_text())
    score = data.get("mythos_score", 0)
    components = data.get("components", {})
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in sorted(components.items())
    )
    html = f"""<!DOCTYPE html>
<html><head><title>Mythos Scoreboard</title>
<style>body{{font-family:sans-serif;margin:2rem}} table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:0.5rem 1rem}}</style></head>
<body><h1>MYTHOS_SCORE: {score:.4f}</h1>
<table><tr><th>Component</th><th>Normalized</th></tr>{rows}</table>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


if __name__ == "__main__":
    render_dashboard(Path("eval/results/latest.json"), Path("eval/dashboards/index.html"))
