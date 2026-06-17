from pathlib import Path

from mythos.autoresearch import load_best_bpb, append_result


def test_results_tsv_roundtrip(tmp_path):
    path = tmp_path / "results.tsv"
    append_result(
        path,
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "experiment": 1,
            "val_bpb": 1.0,
            "mythos_score": 0.15,
            "kept": True,
            "commit": "abc",
            "notes": "{}",
        },
    )
    assert load_best_bpb(path) == 1.0
