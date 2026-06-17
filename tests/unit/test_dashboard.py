"""Dashboard tests."""

from pathlib import Path

from eval.dashboards.render import load_results_tsv, render_dashboard


def test_render_empty_results(tmp_path):
    results = tmp_path / "results.tsv"
    out = tmp_path / "index.html"
    render_dashboard(results, out)
    assert out.exists()
    assert "No autoresearch runs yet" in out.read_text()


def test_render_tsv_rows(tmp_path):
    results = tmp_path / "results.tsv"
    results.write_text(
        "timestamp\texperiment\tval_bpb\tmythos_score\tkept\tcommit\tnotes\n"
        "2026-01-01T00:00:00Z\t1\t2.500000\t0.450000\tTrue\tabc123\tdemo\n"
    )
    out = tmp_path / "index.html"
    render_dashboard(results, out)
    html = out.read_text()
    assert "2.500000" in html
    assert "Mythos Live Demo" in html
    assert len(load_results_tsv(results)) == 1
