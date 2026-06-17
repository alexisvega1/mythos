from mythos.eval.composite import RawScores, compute_mythos_score


def test_mythos_score_lower_bpb_is_better():
    good = compute_mythos_score(RawScores(val_bpb=2.0))
    bad = compute_mythos_score(RawScores(val_bpb=6.0))
    assert good.mythos_score is not None and bad.mythos_score is not None
    assert good.mythos_score > bad.mythos_score


def test_unavailable_metrics_renormalize():
    partial = compute_mythos_score(RawScores(val_bpb=3.0, gsm8k_acc=None))
    assert partial.mythos_score is not None
    assert "gsm8k" in partial.unavailable
