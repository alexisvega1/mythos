from mythos.eval.harness import run_full_eval
from mythos.eval.composite import compute_mythos_score


def test_proxy_eval_runs():
    raw = run_full_eval(limit=3, mode="proxy")
    composite = compute_mythos_score(raw)
    assert 0 <= composite.mythos_score <= 1.5
    assert "swe_bench_verified" in composite.components
