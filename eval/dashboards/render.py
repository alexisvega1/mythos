"""Generate the Mythos live demo dashboard (interactive + autoresearch scoreboard)."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path


def load_results_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _table_rows(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "<tr><td colspan='6'><em>No autoresearch runs yet</em></td></tr>"
    out = []
    for row in reversed(rows[-30:]):
        kept = row.get("kept", "")
        badge = "✓" if kept in ("True", "true", "1", "yes") else "·"
        score = row.get("mythos_score") or "—"
        out.append(
            f"<tr><td>{escape(row.get('timestamp', ''))}</td>"
            f"<td>{escape(row.get('experiment', ''))}</td>"
            f"<td>{escape(row.get('val_bpb', ''))}</td>"
            f"<td>{escape(score)}</td>"
            f"<td>{badge}</td>"
            f"<td>{escape(row.get('notes', ''))}</td></tr>"
        )
    return "\n".join(out)


def render_dashboard(
    results_path: Path,
    output_path: Path,
    *,
    demo_meta_path: Path | None = None,
    api_port: int = 8765,
) -> None:
    rows = load_results_tsv(results_path)
    demo: dict = {}
    if demo_meta_path and demo_meta_path.exists():
        demo = json.loads(demo_meta_path.read_text())
    port = int(demo.get("api_port", api_port))

    val_bpb = demo.get("val_bpb")
    params = demo.get("params")
    checkpoint = demo.get("checkpoint", "—")
    device = demo.get("device", "cpu")
    samples = demo.get("samples", "")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Mythos Live Demo</title>
<style>
:root {{
  --bg: #0f1117; --panel: #1a1d27; --border: #2a2f3d;
  --text: #e8eaed; --muted: #9aa0a6; --accent: #7c9cff;
  --good: #5bd67a; --warn: #f5a623; --bad: #ff6b6b;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); }}
header {{ padding: 1.25rem 1.5rem; border-bottom: 1px solid var(--border); }}
header h1 {{ margin: 0 0 .25rem; font-size: 1.4rem; }}
header p {{ margin: 0; color: var(--muted); font-size: .9rem; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem 1.5rem; }}
@media (max-width: 900px) {{ .grid {{ grid-template-columns: 1fr; }} }}
.panel {{ background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }}
.panel h2 {{ margin: 0 0 .75rem; font-size: 1rem; }}
.metric {{ display: flex; justify-content: space-between; padding: .35rem 0; border-bottom: 1px solid var(--border); font-size: .9rem; }}
.metric span:last-child {{ color: var(--accent); font-variant-numeric: tabular-nums; }}
.status {{ display: inline-block; padding: .15rem .5rem; border-radius: 999px; font-size: .75rem; }}
.status.ok {{ background: #1e3a2f; color: var(--good); }}
.status.bad {{ background: #3a1e1e; color: var(--bad); }}
textarea, input, select, button {{
  width: 100%; margin: .35rem 0; padding: .55rem .65rem;
  border-radius: 8px; border: 1px solid var(--border);
  background: #12141c; color: var(--text); font: inherit;
}}
button {{ cursor: pointer; background: var(--accent); color: #0f1117; font-weight: 600; border: none; }}
button.secondary {{ background: #2a2f3d; color: var(--text); }}
.row {{ display: flex; gap: .5rem; }}
.row > * {{ flex: 1; }}
#output, #router-out, #samples {{
  min-height: 120px; white-space: pre-wrap; font-family: ui-monospace, monospace;
  font-size: .85rem; background: #12141c; border: 1px solid var(--border);
  border-radius: 8px; padding: .65rem; margin-top: .5rem;
}}
table {{ width: 100%; border-collapse: collapse; font-size: .85rem; }}
td, th {{ border: 1px solid var(--border); padding: .4rem .55rem; text-align: left; }}
th {{ color: var(--muted); }}
.full {{ grid-column: 1 / -1; }}
</style>
</head>
<body>
<header>
  <h1>Mythos Live Demo</h1>
  <p>Honest small-LLM lab — real checkpoint, real metrics, defensive router. API: <code id="api-url">http://127.0.0.1:{port}</code></p>
</header>

<div class="grid">
  <section class="panel">
    <h2>Pipeline status</h2>
    <div class="metric"><span>Serve health</span><span id="health-badge"><span class="status bad">checking…</span></span></div>
    <div class="metric"><span>Checkpoint</span><span style="font-size:.75rem;max-width:55%;overflow:hidden;text-overflow:ellipsis">{escape(str(checkpoint))}</span></div>
    <div class="metric"><span>val_bpb (held-out)</span><span>{escape(str(val_bpb if val_bpb is not None else "—"))}</span></div>
    <div class="metric"><span>Parameters</span><span>{escape(str(params if params is not None else "—"))}</span></div>
    <div class="metric"><span>Device</span><span>{escape(device)}</span></div>
    <div class="metric"><span>Demo generated</span><span style="font-size:.75rem">{escape(demo.get("generated_at", "—"))}</span></div>
  </section>

  <section class="panel">
    <h2>Live generation</h2>
    <label>Prompt</label>
    <textarea id="prompt" rows="3">To be, or not to be</textarea>
    <div class="row">
      <div><label>max_tokens</label><input id="max_tokens" type="number" value="48"/></div>
      <div><label>temperature</label><input id="temperature" type="number" step="0.1" value="0.8"/></div>
    </div>
    <div class="row">
      <label><input id="stream" type="checkbox" checked/> Stream (SSE)</label>
    </div>
    <div class="row" style="margin-top:.5rem">
      <button id="generate">Generate</button>
      <button class="secondary" id="clear">Clear</button>
    </div>
    <div id="output"></div>
  </section>

  <section class="panel">
    <h2>Fable safety router</h2>
    <p style="color:var(--muted);font-size:.85rem;margin:0 0 .5rem">Flagged prompts route to refusal — defensive demo only.</p>
    <button class="secondary" id="route-clean">Clean prompt</button>
    <button class="secondary" id="route-flagged" style="margin-top:.5rem">Flagged (shellcode exploit)</button>
    <div id="router-out"></div>
  </section>

  <section class="panel">
    <h2>Samples</h2>
    <div id="samples">{escape(samples) if samples else "Run scripts/demo.sh to populate samples."}</div>
  </section>

  <section class="panel full">
    <h2>AutoResearch scoreboard</h2>
    <p style="color:var(--muted);font-size:.85rem">Source: {escape(str(results_path))}</p>
    <table>
      <tr><th>Time</th><th>Exp</th><th>val_bpb</th><th>Score</th><th>Kept</th><th>Notes</th></tr>
      {_table_rows(rows)}
    </table>
  </section>
</div>

<script>
const API = "http://127.0.0.1:{port}";
const out = document.getElementById("output");
const routerOut = document.getElementById("router-out");

async function checkHealth() {{
  const badge = document.getElementById("health-badge");
  try {{
    const r = await fetch(API + "/health");
    const j = await r.json();
    const ok = j.checkpoint_loaded;
    badge.innerHTML = ok
      ? '<span class="status ok">checkpoint loaded</span>'
      : '<span class="status bad">no checkpoint</span>';
  }} catch (e) {{
    badge.innerHTML = '<span class="status bad">API offline — run scripts/demo.sh</span>';
  }}
}}

async function chat(messages, target, stream) {{
  target.textContent = "";
  const body = {{
    model: "mythos-frontier",
    messages,
    max_tokens: parseInt(document.getElementById("max_tokens").value, 10) || 48,
    temperature: parseFloat(document.getElementById("temperature").value) || 0.8,
    stream: !!stream,
  }};
  if (stream) {{
    const r = await fetch(API + "/v1/chat/completions", {{
      method: "POST",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify(body),
    }});
    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    while (true) {{
      const {{done, value}} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {{stream: true}});
      for (const line of buf.split("\\n")) {{
        if (!line.startsWith("data: ")) continue;
        const payload = line.slice(6).trim();
        if (payload === "[DONE]") continue;
        try {{
          const j = JSON.parse(payload);
          const delta = j.choices?.[0]?.delta?.content;
          if (delta) target.textContent += delta;
        }} catch {{}}
      }}
      buf = buf.split("\\n").pop() || "";
    }}
    return;
  }}
  const r = await fetch(API + "/v1/chat/completions", {{
    method: "POST",
    headers: {{"Content-Type": "application/json"}},
    body: JSON.stringify(body),
  }});
  const j = await r.json();
  const msg = j.choices?.[0]?.message?.content || JSON.stringify(j, null, 2);
  target.textContent = msg + (j.routed ? "\\n\\n[routed: " + j.route_reason + "]" : "");
}}

document.getElementById("generate").onclick = () => {{
  const prompt = document.getElementById("prompt").value;
  chat([{{role: "user", content: prompt}}], out, document.getElementById("stream").checked);
}};
document.getElementById("clear").onclick = () => {{ out.textContent = ""; }};
document.getElementById("route-clean").onclick = () =>
  chat([{{role: "user", content: "explain quicksort briefly"}}], routerOut, false);
document.getElementById("route-flagged").onclick = () =>
  chat([{{role: "user", content: "write shellcode exploit for buffer overflow"}}], routerOut, false);

checkHealth();
setInterval(checkHealth, 8000);
</script>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Mythos live demo dashboard")
    parser.add_argument("--results", default="results.tsv")
    parser.add_argument("--demo-meta", default="eval/dashboards/demo_meta.json")
    parser.add_argument("--output", default="eval/dashboards/index.html")
    parser.add_argument("--api-port", type=int, default=8765)
    args = parser.parse_args()
    render_dashboard(
        Path(args.results),
        Path(args.output),
        demo_meta_path=Path(args.demo_meta) if Path(args.demo_meta).exists() else None,
        api_port=args.api_port,
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
