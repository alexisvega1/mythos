import json
from pathlib import Path

from mythos.eval.composite import compute_mythos_score, raw_from_dict
from mythos.eval.harness import run_full_eval


def test_score_reproducible_proxy():
    raw1 = run_full_eval(limit=5, mode="proxy")
    raw2 = run_full_eval(limit=5, mode="proxy")
    c1 = compute_mythos_score(raw1)
    c2 = compute_mythos_score(raw2)
    assert c1.mythos_score == c2.mythos_score


def test_regression_within_golden_bounds():
    golden_path = Path(__file__).parent / "golden_scores.json"
    golden = json.loads(golden_path.read_text())
    raw = run_full_eval(limit=5, mode="proxy")
    composite = compute_mythos_score(raw)
    max_drop = 0.02
    assert composite.mythos_score >= golden["mythos_score"] - max_drop

    for key, golden_val in golden["raw"].items():
        actual = getattr(raw, key)
        if key == "val_bpb":
            assert actual <= golden_val + max_drop
        else:
            assert actual >= golden_val - max_drop
