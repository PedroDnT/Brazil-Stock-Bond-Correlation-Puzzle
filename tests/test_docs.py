"""
Documentation coherence.

Separate from test_paper_consistency.py because these tests read only the documents —
no generated outputs/, no network — so they run in CI, where prose drift is exactly the
kind of thing nobody notices by hand.

Three failure modes they exist to catch:
  1. The implementation guide describing a pipeline that does not exist. It was once a
     project plan naming return columns (imab, irfm, imas) the code never had.
  2. Documents 01 and 02 reading as findings. They were written before the empirical
     work and several of their expectations were falsified by it.
  3. Counts quoted in prose drifting from the suite. The paper claimed "127 tests: 48
     estimator + 79 paper-consistency" long after all three numbers had moved.
"""

import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

BASE = Path(__file__).parent.parent
PAPER = BASE / "docs" / "04_final_paper.md"
README = BASE / "README.md"
GUIDE = BASE / "docs" / "03_implementation_guide.md"
DOC01 = BASE / "docs" / "01_stock_bond_diversification.md"
DOC02 = BASE / "docs" / "02_quantifying_hidden_correlation_risk.md"
WORKFLOW = BASE / ".github" / "workflows" / "tests.yml"

ALL_DOCS = [PAPER, README, GUIDE, DOC01, DOC02]


@lru_cache(maxsize=None)
def _doc(path):
    return path.read_text(encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# Documents 01 and 02 predate the results
# ═════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("path", [DOC01, DOC02])
def test_pre_study_documents_say_so(path):
    """01 and 02 predate the results. Without the banner they read as findings."""
    head = _doc(path)[:2500]
    assert "written before the empirical work" in head, path.name
    assert "04_final_paper.md" in head, f"{path.name} must point at the paper"


# ═════════════════════════════════════════════════════════════════════════════
# The guide describes the pipeline that exists
# ═════════════════════════════════════════════════════════════════════════════
# Series this pipeline has never ingested. Checked case-insensitively and with the
# hyphens the repo's own prose uses, because "IMA-B" is the form anyone would
# actually write and `"imab" in "IMA-B"` is False.
GHOST_SERIES = ["imab", "ima-b", "irfm", "irf-m", "imas", "ima-s", "mvgarch", "mv-garch"]


def test_implementation_guide_describes_the_pipeline_that_exists():
    """The guide was once a project plan naming columns the pipeline never had."""
    txt = _doc(GUIDE).lower()
    for ghost in GHOST_SERIES:
        assert ghost not in txt, f"guide references '{ghost}', which is not in this pipeline"
    for planning in ("week 1", "week 2", "day 1", "buy trigger", "priority queue"):
        assert planning not in txt, f"guide still reads as a schedule ('{planning}')"


def test_guide_names_the_real_target_tenors():
    """Pure text — deliberately not gated on the dataset, so it runs in CI."""
    assert "NTN-B 5y, LTN 2y, NTN-F 10y" in _doc(GUIDE)


def test_the_real_return_columns_are_what_the_guide_says():
    """The data half of the same claim. Skips without a built dataset; the text
    assertion above does not, so deleting the guide's tenor line still fails CI."""
    pd = pytest.importorskip("pandas")
    master = BASE / "data" / "processed" / "master_returns.csv"
    if not master.exists():
        pytest.skip("master not built — run python3 src/fetch.py")
    cols = pd.read_csv(master, nrows=1).columns
    for col in ("ntnb", "ltn", "ntnf", "lft"):
        assert col in cols, f"{col} missing from master_returns.csv"


# ═════════════════════════════════════════════════════════════════════════════
# The open-work list
# ═════════════════════════════════════════════════════════════════════════════
def test_open_work_is_documented_in_one_place_and_linked():
    """The gaps must be findable from the README and the paper, not only in chat."""
    assert "## Known limitations and open work" in _doc(GUIDE)
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


# ═════════════════════════════════════════════════════════════════════════════
# Counts quoted in prose
# ═════════════════════════════════════════════════════════════════════════════
def _ci_files():
    """The files CI actually runs, read from the workflow rather than duplicated.

    A hand-copied second list drifts: add a file to the workflow and the '71
    network-free tests' figure asserted into three documents silently stays right.
    """
    command = "\n".join(l for l in _doc(WORKFLOW).splitlines()
                        if not l.lstrip().startswith("#"))
    found = re.findall(r"tests/\w+\.py", command)
    assert found, "no test files found in the workflow — has the run line changed?"
    return sorted(set(found))


@lru_cache(maxsize=None)
def _collection():
    """{filename: n_tests} from one `--collect-only` pass over the whole suite."""
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--collect-only",
                        "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=BASE)
    # Do NOT skip on failure. pytest still prints "N tests collected" when a module
    # errors during collection, so a silent pass here would certify counts taken
    # over a suite that could not be collected.
    assert r.returncode == 0, (
        f"collection failed (rc={r.returncode}) — fix the suite, not the documents\n"
        f"{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    counts = {}
    for line in r.stdout.splitlines():
        if "::" in line:
            counts[line.split("::")[0]] = counts.get(line.split("::")[0], 0) + 1
    assert counts, f"parsed no node ids from collection output:\n{r.stdout[-2000:]}"
    return counts


def _quotes(text, n, unit):
    """Word-anchored, so '155 tests' is not satisfied by '1155 tests'."""
    return re.search(rf"(?<!\d){n} {re.escape(unit)}", text) is not None


def test_documents_quote_the_real_test_totals():
    """
    The paper claimed '127 tests: 48 estimator + 79 paper-consistency' long after all
    three numbers had moved. Nothing pinned them, so nothing noticed.
    """
    counts = _collection()
    total = sum(counts.values())
    ci = sum(counts[f] for f in _ci_files())
    consistency = counts["tests/test_paper_consistency.py"]
    assert total == ci + consistency, (counts, _ci_files())

    for path, label in [(README, "README"), (GUIDE, "guide"), (PAPER, "paper")]:
        assert _quotes(_doc(path), total, "tests"), \
            f"{label} does not state the real total ({total})"

    assert _quotes(_doc(README), ci, "network-free tests")
    assert _quotes(_doc(GUIDE), ci, "network-free tests")
    assert f"{ci} network-free + {consistency} paper-consistency" in _doc(PAPER)
    assert _quotes(_doc(README), consistency, "consistency")


def test_the_per_file_counts_in_the_trees_are_real():
    """README and the guide both print a per-file breakdown; nothing pinned it."""
    counts = _collection()
    for path in (README, GUIDE):
        txt = _doc(path)
        for filename, n in counts.items():
            stem = Path(filename).name
            if stem not in txt:
                continue
            block = txt[txt.index(stem):txt.index(stem) + 120]
            m = re.search(r"#\s*(\d+) ", block)
            assert m, f"{path.name}: no count beside {stem}"
            assert int(m.group(1)) == n, \
                f"{path.name} says {m.group(1)} tests for {stem}, actual {n}"


# ═════════════════════════════════════════════════════════════════════════════
# Gitignored paths must not be tracked
# ═════════════════════════════════════════════════════════════════════════════
# These have been removed twice and re-added by tooling each time, silently,
# because .gitignore does not stop a `git add -f`. A failing build is the only
# signal that survives an automated re-add.
IGNORED_DIRS = [".claude", ".codex", ".agents", ".vscode"]


def test_gitignored_agent_configs_are_not_tracked():
    listed = subprocess.run(["git", "ls-files", *IGNORED_DIRS],
                            capture_output=True, text=True, cwd=BASE)
    if listed.returncode != 0:
        pytest.skip("not a git checkout")
    tracked = [f for f in listed.stdout.split() if f]
    assert not tracked, (
        "these paths are in .gitignore but tracked anyway — something force-added "
        f"them:\n  " + "\n  ".join(tracked))


@pytest.mark.parametrize("directory", IGNORED_DIRS)
def test_gitignore_still_lists_the_agent_config_dirs(directory):
    """The guard above is only meaningful while .gitignore still excludes them."""
    assert f"{directory}/" in _doc(BASE / ".gitignore"), directory
