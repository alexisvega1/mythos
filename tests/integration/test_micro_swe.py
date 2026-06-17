from mythos.agents.swe import (
    MICRO_TASKS,
    evaluate_swe_proxy,
    noop_solver,
    oracle_solver,
    run_micro_swe_eval,
)


def test_oracle_solver_passes_all():
    """A correct solver scores 1.0 — the execution oracle really runs the fixes."""
    res = run_micro_swe_eval(oracle_solver)
    assert res["n"] == len(MICRO_TASKS)
    assert res["micro_swe_pass_at_1"] == 1.0


def test_noop_solver_fails_buggy():
    """Unchanged buggy code scores 0.0 — the harness genuinely discriminates."""
    res = run_micro_swe_eval(noop_solver)
    assert res["micro_swe_pass_at_1"] == 0.0


def test_real_swe_bench_still_unavailable():
    """No fabricated SWE-bench numbers; real benchmark stays unavailable."""
    assert evaluate_swe_proxy()["swe_bench_verified_pass_at_1"] is None
