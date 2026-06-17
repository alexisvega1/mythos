"""Generate a local scoreboard from autoresearch results.tsv (real runs only)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_results_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def render_dashboard(results_path: Path, output_path: Path) -> None:
    rows = load_results_tsv(results_path)
    table_rows = ""
    for row in reversed(rows[-50:]):
        kept = row.get("kept", "")
        badge = "✓" if kept in ("True", "true", "1", "yes") else "·"
        score = row.get("mythos_score") or "—"
        table_rows += (
            f"<tr><td>{row.get('timestamp', '')}</td>"
            f"<td>{row.get('experiment', '')}</td>"
            f"<td>{row.get('val_bpb', '')}</td>"
            f"<td>{score}</td>"
            f"<td>{badge}</td>"
            f"<td>{row.get('notes', '')}</td></tr>\n"
        )
    if not table_rows:
        table_rows = "<tr><td colspan='6'><em>No runs yet — start autoresearch to populate results.tsv</em></td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><title>Mythos AutoResearch</title>
<meta charset="utf-8"/>
<style>
body{{font-family:system-ui,sans-serif;margin:2rem;max-width:960px}}
table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:0.5rem 0.75rem;text-align:left}}
th{{background:#f5f5f5}}
.muted{{color:#666;font-size:0.9rem}}
</style></head>
<body>
<h1>Mythos AutoResearch</h1>
<p class="muted">Source: {results_path} — only rows from real training runs (no fabricated metrics).</p>
<table>
<tr><th>Timestamp</th><th>Exp</th><th>val_bpb</th><th>MYTHOS_SCORE</th><th>Kept</th><th>Notes</th></tr>
{table_rows}
</table>
</body></html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render autoresearch dashboard")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--output", default="eval/dashboards/index.html")
    args = parser.parse_args()
    render_dashboard(Path(args.results), Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
