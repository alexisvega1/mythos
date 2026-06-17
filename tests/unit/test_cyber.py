from mythos.agents.cyber import ExploitSandbox, score_flags


def test_exploit_sandbox_crash_flags():
    sandbox = ExploitSandbox("v8-cve-2024-001")
    flags = sandbox.run_pov("trigger crash")
    assert flags["coverage"] is True
    assert flags["crash"] is True


def test_score_flags_weighted():
    score = score_flags({"coverage": True, "crash": True, "ace": True})
    assert 0 < score <= 1.0
