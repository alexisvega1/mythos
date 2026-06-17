from mythos.eval.composite import RawScores, compute_mythos_score


def test_mythos_score_increases_with_swe():
    low = compute_mythos_score(RawScores(swe_bench_verified_pass_at_1=0.1))
    high = compute_mythos_score(RawScores(swe_bench_verified_pass_at_1=0.7))
    assert high.mythos_score > low.mythos_score


def test_val_bpb_efficiency_component():
    good = compute_mythos_score(RawScores(val_bpb=0.5))
    bad = compute_mythos_score(RawScores(val_bpb=1.4))
    assert good.components["val_bpb"] > bad.components["val_bpb"]
