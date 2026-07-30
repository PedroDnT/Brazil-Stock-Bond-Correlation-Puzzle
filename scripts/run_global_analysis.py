"""
run_global_analysis.py — Test the convergence hypothesis on a matched panel.

    python3 scripts/run_global_analysis.py

The paper's Section 9 claims advanced economies are converging toward Brazil's
stock-bond correlation regime. That claim cannot be settled from Brazilian data,
so this script builds the same asset definitions and the same construction for
five countries and tests it directly.

Writes CSVs to outputs/ (prefix `tbl_global_`) and prints a summary.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))

import metrics as M                                              # noqa: E402
from fetch import load_master                                     # noqa: E402
from global_data import (build_global_panel, validate_panel,      # noqa: E402
                         validate_against_pu_construction,
                         ALL_COUNTRIES, IMF_BREAK)

OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

BENCH = "BR"
DM = ["US", "DE", "JP", "GB"]
NAME = {cc: spec["name"] for cc, spec in ALL_COUNTRIES.items()}


def hdr(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


def save(df, name):
    df.to_csv(OUT / name)
    print(f"  -> outputs/{name}")


# ─────────────────────────────────────────────────────────────────────────────
hdr("1. Matched panel")
master = load_master()
panel = build_global_panel(master=master)
validate_panel(panel)
xcheck = validate_against_pu_construction(master=master)

pre = panel[panel.index <= IMF_BREAK]
post = panel[panel.index > IMF_BREAK]
print(f"\n  pre-break : {pre.index.min().date()} to {pre.index.max().date()}  "
      f"({len(pre)} months)")
print(f"  post-break: {post.index.min().date()} to {post.index.max().date()}  "
      f"({len(post)} months)")
print(f"  break at {IMF_BREAK} — the turning point the IMF note identifies")


# ─────────────────────────────────────────────────────────────────────────────
hdr("2. Stock-bond correlation by country and period")
rows = []
for cc in DM + [BENCH]:
    full = M.corr_with_ci(panel[f"{cc}_eq"], panel[f"{cc}_bd"])
    p0 = M.corr_with_ci(pre[f"{cc}_eq"], pre[f"{cc}_bd"])
    p1 = M.corr_with_ci(post[f"{cc}_eq"], post[f"{cc}_bd"])
    t = M.bootstrap_corr_diff(post[f"{cc}_eq"], post[f"{cc}_bd"],
                              pre[f"{cc}_eq"], pre[f"{cc}_bd"],
                              n_boot=2000, block=6, seed=2)
    rows.append({
        "Country": NAME[cc],
        "rho full": round(full["rho"], 3),
        "rho pre-2020": round(p0["rho"], 3),
        "CI pre": f"[{p0['lo']:+.3f},{p0['hi']:+.3f}]",
        "rho post-2020": round(p1["rho"], 3),
        "CI post": f"[{p1['lo']:+.3f},{p1['hi']:+.3f}]",
        "shift": round(p1["rho"] - p0["rho"], 3),
        "p(shift)": round(t["p"], 3),
        "shifted 5%": "yes" if t["p"] < 0.05 else "no",
    })
corr_tbl = pd.DataFrame(rows).set_index("Country")
save(corr_tbl, "tbl_global_correlations.csv")
print(corr_tbl.to_string())

print("\n  Reading: a negative pre-2020 correlation that turns positive after 2019 is")
print("  the IMF's finding. Brazil's row is the comparison — if it shows no shift,")
print("  its regime is the stable one that the others may be moving toward.")


# ─────────────────────────────────────────────────────────────────────────────
hdr("3. The convergence test")
print("  H0: each advanced economy's correlation shift equals Brazil's.")
print("  DiD = [rho_post(c) - rho_pre(c)] - [rho_post(BR) - rho_pre(BR)]")
print("  DiD > 0 and significant => c moved toward Brazil by more than Brazil moved.\n")

rho_pre_br = M.corr_with_ci(pre[f"{BENCH}_eq"], pre[f"{BENCH}_bd"])["rho"]
rho_post_br = M.corr_with_ci(post[f"{BENCH}_eq"], post[f"{BENCH}_bd"])["rho"]

rows = []
for cc in DM:
    r0 = M.corr_with_ci(pre[f"{cc}_eq"], pre[f"{cc}_bd"])["rho"]
    r1 = M.corr_with_ci(post[f"{cc}_eq"], post[f"{cc}_bd"])["rho"]
    gap0, gap1 = abs(rho_pre_br - r0), abs(rho_post_br - r1)
    d = M.bootstrap_did(
        (pre[f"{cc}_eq"], pre[f"{cc}_bd"]), (post[f"{cc}_eq"], post[f"{cc}_bd"]),
        (pre[f"{BENCH}_eq"], pre[f"{BENCH}_bd"]), (post[f"{BENCH}_eq"], post[f"{BENCH}_bd"]),
        n_boot=2000, block=6, seed=3)
    rows.append({
        "Country": NAME[cc],
        "gap to BR, pre": round(gap0, 3),
        "gap to BR, post": round(gap1, 3),
        "gap narrowed": "yes" if gap1 < gap0 else "no",
        "narrowing": round(gap0 - gap1, 3),
        "DiD": round(d["did"], 3),
        "boot SE": round(d["boot_se"], 3),
        "p": round(d["p"], 3),
        "converged 5%": "yes" if (d["p"] < 0.05 and gap1 < gap0) else "no",
    })
conv = pd.DataFrame(rows).set_index("Country")
save(conv, "tbl_global_convergence.csv")
print(conv.to_string())

n_narrow = (conv["gap narrowed"] == "yes").sum()
n_sig = (conv["converged 5%"] == "yes").sum()
print(f"\n  Gap to Brazil narrowed in {n_narrow}/4 advanced economies.")
print(f"  Narrowing is statistically significant in {n_sig}/4.")


# ─────────────────────────────────────────────────────────────────────────────
hdr("4. Is Brazil's regime the stable one?")
rows = []
for cc in DM + [BENCH]:
    r0 = M.corr_with_ci(pre[f"{cc}_eq"], pre[f"{cc}_bd"])
    r1 = M.corr_with_ci(post[f"{cc}_eq"], post[f"{cc}_bd"])
    rows.append({
        "Country": NAME[cc],
        "pre-2020 sign": "negative" if r0["hi"] < 0 else "positive" if r0["lo"] > 0 else "indistinct",
        "post-2020 sign": "negative" if r1["hi"] < 0 else "positive" if r1["lo"] > 0 else "indistinct",
        "ever significantly negative": "yes" if (r0["hi"] < 0 or r1["hi"] < 0) else "no",
    })
signs = pd.DataFrame(rows).set_index("Country")
save(signs, "tbl_global_sign_regimes.csv")
print(signs.to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("5. Rolling 60-month correlation")
W = 60
roll = pd.DataFrame({
    NAME[cc]: panel[f"{cc}_eq"].rolling(W).corr(panel[f"{cc}_bd"])
    for cc in DM + [BENCH]
}).dropna(how="all")
save(roll.round(4), "tbl_global_rolling_correlation.csv")
print(f"  {W}-month rolling correlation, {roll.index.min().date()} to {roll.index.max().date()}")
print("\n  Latest values:")
print(roll.dropna().iloc[-1].round(3).to_string())
print("\n  Minimum over the sample (how negative each market ever got):")
print(roll.min().round(3).to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("6. Construction cross-check (licenses the yield-based method)")
xc = pd.DataFrame([{
    "yield-based vs PU-based, rho": round(xcheck["rho_constructions"], 4),
    "Ibov x bond, yield-based": round(xcheck["rho_yield_based"], 3),
    "Ibov x bond, PU-based": round(xcheck["rho_pu_based"], 3),
    "absolute difference": round(abs(xcheck["rho_yield_based"] - xcheck["rho_pu_based"]), 4),
    "n months": xcheck["n"],
}]).T.rename(columns={0: "value"})
save(xc, "tbl_global_construction_check.csv")
print(xc.to_string())

print("\n" + "=" * 78)
print("  Global analysis complete")
print("=" * 78)
