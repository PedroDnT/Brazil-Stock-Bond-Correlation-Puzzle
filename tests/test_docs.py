"""
Documentation coherence.

Separate from test_paper_consistency.py because these tests read only the documents —
no generated outputs/, no network — so they run in CI, where prose drift is exactly the
kind of thing nobody notices by hand.

Two failure modes they exist to catch:
  1. The implementation guide describing a pipeline that does not exist. It was once a
     project plan naming return columns (imab, irfm, imas) the code never had.
  2. Documents 01 and 02 reading as findings. They were written before the empirical
     work and several of their expectations were falsified by it.
"""

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

BASE = Path(__file__).parent.parent
PAPER = BASE / "docs" / "04_final_paper.md"
README = BASE / "README.md"
GUIDE = BASE / "docs" / "03_implementation_guide.md"
DOC01 = BASE / "docs" / "01_stock_bond_diversification.md"
DOC02 = BASE / "docs" / "02_quantifying_hidden_correlation_risk.md"


def _doc(path):
    return path.read_text(encoding="utf-8")

@pytest.mark.parametrize("path", [DOC01, DOC02])
def test_pre_study_documents_say_so(path):
    """01 and 02 predate the results. Without the banner they read as findings."""
    head = _doc(path)[:2500]
    assert "written before the empirical work" in head, path.name
    assert "04_final_paper.md" in head, f"{path.name} must point at the paper"


def test_implementation_guide_describes_the_pipeline_that_exists():
    """The guide was once a project plan naming columns the pipeline never had."""
    txt = _doc(GUIDE)
    for ghost in ("imab", "irfm", "imas", "mvgarch"):
        assert ghost not in txt, f"guide references '{ghost}', which is not in this pipeline"
    for planning in ("Week 1", "Week 2", "Day 1", "Buy trigger", "Priority queue"):
        assert planning not in txt, f"guide still reads as a schedule ('{planning}')"


def test_implementation_guide_names_the_real_return_columns():
    master = pd.read_csv(BASE / "data" / "processed" / "master_returns.csv", nrows=1) \
        if (BASE / "data" / "processed" / "master_returns.csv").exists() else None
    if master is None:
        pytest.skip("master not built")
    txt = _doc(GUIDE)
    for col in ("ntnb", "ltn", "ntnf", "lft"):
        assert col in master.columns, f"{col} missing from master — guide is out of date"
    assert "NTN-B 5y, LTN 2y, NTN-F 10y" in txt


def test_open_work_is_documented_in_one_place_and_linked():
    """The gaps must be findable from the README and the paper, not only in chat."""
    guide = _doc(GUIDE)
    assert "## Known limitations and open work" in guide

    anchor = "03_implementation_guide.md#known-limitations-and-open-work"
    assert anchor in _doc(README), "README does not link the open-work list"
    assert anchor in _doc(PAPER), "paper's Limitations does not link the open-work list"


@pytest.mark.parametrize("gap", [
    "sovereign-credit channel has no single measure",
    "Regime boundaries are imposed",
    "cross-country break date is imposed",
    "remain unreproduced",
    "consistency tests do not run in CI",
])
def test_each_known_gap_is_stated(gap):
    assert gap in _doc(GUIDE), f"open-work list no longer states: {gap}"


# ── Test counts quoted in prose ──────────────────────────────────────────────
def _collected(paths):
    """Ask pytest how many tests it collects, so the docs cannot drift from reality."""
    import subprocess, sys, re
    r = subprocess.run([sys.executable, "-m", "pytest", *paths, "-q", "--collect-only",
                        "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=BASE)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    if not m:
        pytest.skip(f"could not collect: {r.stdout[-300:]}")
    return int(m.group(1))


CI_FILES = ["tests/test_metrics.py", "tests/test_global_data.py",
            "tests/test_sovereign.py", "tests/test_docs.py"]


def test_documents_quote_the_real_test_totals():
    """
    The paper claimed '127 tests: 48 estimator + 79 paper-consistency' long after all
    three numbers had moved. Nothing pinned them, so nothing noticed.
    """
    total = _collected(["tests/"])
    ci = _collected(CI_FILES)
    consistency = _collected(["tests/test_paper_consistency.py"])

    for path, label in [(README, "README"), (GUIDE, "guide"), (PAPER, "paper")]:
        txt = _doc(path)
        assert f"{total} tests" in txt, f"{label} does not state the real total ({total})"

    assert f"{ci} network-free tests" in _doc(README)
    assert f"{ci} network-free tests" in _doc(GUIDE)
    assert f"{ci} network-free + {consistency} paper-consistency" in _doc(PAPER)
    assert f"{consistency} tests" in _doc(README)
