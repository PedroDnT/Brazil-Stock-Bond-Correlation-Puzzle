"""
run_analysis.py — Produce every table the paper cites, headlessly.

    python3 scripts/run_analysis.py

Writes CSVs to outputs/ and prints a summary. The notebooks render the same
quantities as figures; this script exists so the paper's numbers can be
regenerated and diffed without executing notebooks.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / "src"))

import metrics as M                                    # noqa: E402
from fetch import load_master, CRISES, REGIMES         # noqa: E402

OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

RET  = ["ibov", "ntnb", "ltn", "ntnf", "lft"]
BOND = ["ntnb", "ltn", "ntnf", "lft"]
LAB  = {"ibov": "Ibovespa", "ntnb": "NTN-B 5y", "ltn": "LTN 2y",
        "ntnf": "NTN-F 10y", "lft": "LFT 1y", "lft_long": "LFT long"}

master = load_master()
tables = {}


def hdr(t):
    print("\n" + "=" * 78)
    print(f"  {t}")
    print("=" * 78)


def save(df, name):
    df.to_csv(OUT / name)
    tables[name] = df
    print(f"  -> outputs/{name}")


# ─────────────────────────────────────────────────────────────────────────────
hdr("1. Sample")
d = master[RET].dropna()
print(f"  master        : {master.shape[0]:,} rows, {master.index[0].date()} to {master.index[-1].date()}")
print(f"  complete cases: {len(d):,} ({d.index[0].date()} to {d.index[-1].date()})")

rows = []
for c in RET:
    s = master[c].dropna()
    rf = master.loc[s.index, "cdi_level"].mean()
    ann = s.mean() * 252 * 100
    vol = s.std() * np.sqrt(252) * 100
    px = np.exp(s.cumsum())
    rows.append({
        "Asset": LAB[c], "Obs": len(s),
        "Ann ret%": round(ann, 2), "Ann vol%": round(vol, 2),
        "Sharpe (over CDI)": round((ann - rf) / vol, 3) if vol > 0 else np.nan,
        "Skew": round(float(pd.Series(s).skew()), 3),
        "Exc kurt": round(float(pd.Series(s).kurt()), 2),
        "Max DD%": round(float((px / px.cummax() - 1).min() * 100), 2),
    })
save(pd.DataFrame(rows).set_index("Asset"), "tbl_summary_stats.csv")
print(pd.DataFrame(rows).set_index("Asset").to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("2. Full-sample correlations with 95% CIs")
rows = []
for c in BOND:
    r = M.corr_with_ci(master["ibov"], master[c])
    rows.append({"Pair": f"Ibovespa x {LAB[c]}", "n": r["n"], "rho": round(r["rho"], 3),
                 "CI lo": round(r["lo"], 3), "CI hi": round(r["hi"], 3),
                 "p": f"{r['p']:.1e}", "sig 5%": "yes" if r["sig"] else "no"})
full = pd.DataFrame(rows).set_index("Pair")
save(full, "tbl_full_sample_correlations.csv")
print(full.to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("3. Return frequency robustness (Ibovespa x each bond)")
rows = []
for lab, rule in [("daily", None), ("weekly", "W-FRI"), ("monthly", "ME"), ("quarterly", "QE")]:
    for c in BOND:
        x = master[["ibov", c]].dropna()
        if rule:
            x = x.resample(rule).sum()
            x = x[(x != 0).any(axis=1)]
        r = M.corr_with_ci(x["ibov"], x[c])
        rows.append({"Frequency": lab, "Bond": LAB[c], "n": r["n"],
                     "rho": round(r["rho"], 3), "CI lo": round(r["lo"], 3),
                     "CI hi": round(r["hi"], 3), "sig 5%": "yes" if r["sig"] else "no"})
freq = pd.DataFrame(rows)
save(freq.set_index(["Frequency", "Bond"]), "tbl_frequency_robustness.csv")
print(freq.pivot(index="Bond", columns="Frequency", values="rho")
          [["daily", "weekly", "monthly", "quarterly"]].to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("4. Regime-split correlations with CIs")
rows = []
for name, (s, e) in list(REGIMES.items()) + [("Full sample", (str(master.index[0].date()),
                                                              str(master.index[-1].date())))]:
    sub = master[(master.index >= s) & (master.index <= e)]
    row = {"Regime": name, "n": len(sub[["ibov", "ntnb"]].dropna())}
    for c in BOND:
        r = M.corr_with_ci(sub["ibov"], sub[c])
        row[LAB[c]] = round(r["rho"], 3)
        row[f"{LAB[c]} CI"] = f"[{r['lo']:+.3f},{r['hi']:+.3f}]"
        row[f"{LAB[c]} sig"] = "*" if r["sig"] else ""
    rows.append(row)
reg = pd.DataFrame(rows).set_index("Regime")
save(reg, "tbl_regime_correlations.csv")
print(reg[["n"] + [LAB[c] for c in BOND]].to_string())
print("\n  95% CIs (Ibovespa x NTN-B):")
for _, r in reg.iterrows():
    print(f"    {r.name:22s} {r['NTN-B 5y']:+.3f}  {r['NTN-B 5y CI']} {r['NTN-B 5y sig']}")

# is any regime's correlation actually different from the calmest one?
hdr("4b. Are regime differences statistically distinguishable?")
base = "Lula Boom"
bs, be = REGIMES[base]
b_sub = master[(master.index >= bs) & (master.index <= be)][["ibov", "ntnb"]].dropna()
rows = []
for name, (s, e) in REGIMES.items():
    if name == base:
        continue
    sub = master[(master.index >= s) & (master.index <= e)][["ibov", "ntnb"]].dropna()
    t = M.bootstrap_corr_diff(sub["ibov"], sub["ntnb"], b_sub["ibov"], b_sub["ntnb"],
                              n_boot=1500, block=21, seed=1)
    rows.append({"Regime": name, f"rho - rho({base})": round(t["diff"], 3),
                 "boot SE": round(t["boot_se"], 3), "p": round(t["p"], 3),
                 "differs 5%": "yes" if t["p"] < 0.05 else "no"})
diff = pd.DataFrame(rows).set_index("Regime")
save(diff, "tbl_regime_difference_tests.csv")
print(diff.to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("5. Conditional tail correlations (equity stress)")
rows = []
q10 = master["ibov"].quantile(0.10)
q25 = master["ibov"].quantile(0.25)
for c in BOND:
    p = master[["ibov", c]].dropna()
    full_r = M.corr_with_ci(p["ibov"], p[c])
    s10 = p[p["ibov"] <= q10]
    s25 = p[p["ibov"] <= q25]
    r10 = M.corr_with_ci(s10["ibov"], s10[c])
    r25 = M.corr_with_ci(s25["ibov"], s25[c])
    rows.append({
        "Bond": LAB[c],
        "rho full": round(full_r["rho"], 3),
        "rho|Q25": round(r25["rho"], 3), "CI|Q25": f"[{r25['lo']:+.3f},{r25['hi']:+.3f}]",
        "rho|Q10": round(r10["rho"], 3), "CI|Q10": f"[{r10['lo']:+.3f},{r10['hi']:+.3f}]",
        "n|Q10": r10["n"],
        "delta": round(r10["rho"] - full_r["rho"], 3),
    })
cond = pd.DataFrame(rows).set_index("Bond")
save(cond, "tbl_conditional_correlations.csv")
print(cond.to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("6. Forbes-Rigobon: is the crisis correlation surge real?")
calm = master[master["crisis"] == "Calm"]
assert len(calm) > 1000, f"calm window is empty ({len(calm)} rows) — check the crisis label"
rows = []
for cname, (s, e) in CRISES.items():
    k = master[(master.index >= s) & (master.index <= e)]
    for c in ["ntnb", "ltn"]:
        o = M.forbes_rigobon(calm["ibov"], calm[c], k["ibov"], k[c])
        rows.append({"Crisis": cname, "Bond": LAB[c], "n crisis": o["n_crisis"],
                     "rho calm": round(o["rho_calm"], 3),
                     "rho crisis (raw)": round(o["rho_crisis"], 3),
                     "vol ratio": round(1 + o["delta"], 2),
                     "rho adjusted": round(o["rho_adj"], 3),
                     "contagion": {True: "yes", False: "no", None: "n/a"}[o["contagion"]]})
fr = pd.DataFrame(rows).set_index(["Crisis", "Bond"])
save(fr, "tbl_forbes_rigobon.csv")
print(fr.to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("7. DCC-GARCH with standard errors")
from arch import arch_model                                        # noqa: E402

std_resid = {}
print("  GARCH(1,1) stage 1:")
for c in RET:
    s = master[c].dropna() * 100
    res = arch_model(s, vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    std_resid[c] = res.resid / res.conditional_volatility
    p = res.params
    print(f"    {LAB[c]:11s} alpha={p.get('alpha[1]',np.nan):.4f} "
          f"beta={p.get('beta[1]',np.nan):.4f} persist={p.get('alpha[1]',0)+p.get('beta[1]',0):.4f}")

rows, dcc_paths = [], {}
print("\n  DCC stage 2:")
for c in BOND:
    f = M.fit_dcc(std_resid["ibov"], std_resid[c])
    dcc_paths[c] = f["rho"]
    rows.append({"Pair": f"Ibovespa x {LAB[c]}", "a": round(f["a"], 4),
                 "SE(a)": round(f["se_a"], 4), "t(a)": round(f["t_a"], 2),
                 "b": round(f["b"], 4), "SE(b)": round(f["se_b"], 4),
                 "persistence": round(f["persistence"], 4),
                 "mean rho": round(f["rho"].mean(), 3),
                 "sd rho": round(f["rho"].std(), 3),
                 "min rho": round(f["rho"].min(), 3),
                 "max rho": round(f["rho"].max(), 3),
                 "time-varying (a>0, 5%)": ("not identified" if not f["identified"]
                                            else "yes" if f["t_a"] > 1.96 else "no")})
    print(f"    {LAB[c]:11s} a={f['a']:.4f} (t={f['t_a']:.2f})  b={f['b']:.4f}  "
          f"rho: mean={f['rho'].mean():+.3f} sd={f['rho'].std():.3f} "
          f"range [{f['rho'].min():+.3f},{f['rho'].max():+.3f}]")
dcc = pd.DataFrame(rows).set_index("Pair")
save(dcc, "tbl_dcc_parameters.csv")

rows = []
for cname, (s, e) in list(CRISES.items()) + [("Full sample", ("2004-01-01", "2026-12-31"))]:
    row = {"Period": cname}
    for c in BOND:
        r = dcc_paths[c]
        m_ = r[(r.index >= s) & (r.index <= e)]
        row[LAB[c]] = round(m_.mean(), 3) if len(m_) else np.nan
    rows.append(row)
dcc_c = pd.DataFrame(rows).set_index("Period")
save(dcc_c, "tbl_dcc_crisis_correlations.csv")
print("\n" + dcc_c.to_string())
pd.DataFrame(dcc_paths).to_csv(OUT / "dcc_rho_paths.csv")


# ─────────────────────────────────────────────────────────────────────────────
hdr("8. Copulas (corrected densities)")
rows = []
for c in BOND:
    p = master[["ibov", c]].dropna()
    u = M.pseudo_obs(p)
    uu, vv = u["ibov"].values, u[c].values
    fits = M.fit_all_copulas(uu, vv)
    emp = M.tail_dependence_empirical(uu, vv, q=0.05)
    best = fits[0]
    for f in fits:
        rows.append({"Pair": f"Ibovespa x {LAB[c]}", "Copula": f["family"],
                     "logL": round(f["ll"], 1), "AIC": round(f["AIC"], 1),
                     "param": f["param"], "lambda_L": round(f["lambda_L"], 4),
                     "lambda_U": round(f["lambda_U"], 4),
                     "best": "*" if f is best else ""})
    print(f"  {LAB[c]:11s} best={best['family']:10s} ({best['param']})  "
          f"lam_L={best['lambda_L']:.4f} lam_U={best['lambda_U']:.4f}   "
          f"empirical 5% tail: lam_L={emp['lambda_L']:.3f} lam_U={emp['lambda_U']:.3f} "
          f"(indep=0.050); co-crash obs {emp['n_co_lower']} vs {emp['expected_indep']:.0f} expected")
cop = pd.DataFrame(rows).set_index(["Pair", "Copula"])
save(cop, "tbl_copula_fit.csv")

rows = []
for c in BOND:
    p = master[["ibov", c]].dropna()
    u = M.pseudo_obs(p)
    for q in (0.01, 0.05, 0.10):
        e = M.tail_dependence_empirical(u["ibov"].values, u[c].values, q=q)
        rows.append({"Bond": LAB[c], "q": q, "lambda_L": round(e["lambda_L"], 4),
                     "lambda_U": round(e["lambda_U"], 4), "indep benchmark": q,
                     "co-crash obs": e["n_co_lower"],
                     "expected if indep": round(e["expected_indep"], 1)})
save(pd.DataFrame(rows).set_index(["Bond", "q"]), "tbl_tail_dependence.csv")


# ─────────────────────────────────────────────────────────────────────────────
hdr("9. Crisis P&L — total return and excess over CDI")
PORTFOLIOS = {
    "60/40 (Ibov+NTN-B)":    {"ibov": 0.60, "ntnb": 0.40},
    "Diversified (4 asset)": {"ibov": 0.40, "ntnb": 0.30, "ltn": 0.15, "lft": 0.15},
    "Equity heavy (80/20)":  {"ibov": 0.80, "ntnb": 0.20},
    "Bond heavy (20/80)":    {"ibov": 0.20, "ntnb": 0.50, "ltn": 0.20, "lft": 0.10},
    "LFT only (cash)":       {"lft": 1.00},
}
tot, exc = {}, {}
for pname, w in PORTFOLIOS.items():
    rt, re_ = {}, {}
    for cname, (s, e) in CRISES.items():
        sub = master[(master.index >= s) & (master.index <= e)]
        port = sum(wt * sub[c] for c, wt in w.items())
        rt[cname] = round((np.exp(port.sum()) - 1) * 100, 1)
        re_[cname] = round((np.exp(port.sum()) - np.exp(sub["cdi_ret"].sum())) * 100, 1)
    tot[pname], exc[pname] = rt, re_
tot_df, exc_df = pd.DataFrame(tot).T, pd.DataFrame(exc).T
save(tot_df, "tbl_scenario_pnl_total.csv")
save(exc_df, "tbl_scenario_pnl_excess_cdi.csv")
print("  Total return (%):");        print(tot_df.to_string())
print("\n  Excess over CDI (pp):");  print(exc_df.to_string())

rows = {}
for cname, (s, e) in CRISES.items():
    sub = master[(master.index >= s) & (master.index <= e)]
    r = {LAB[c]: round((np.exp(sub[c].sum()) - 1) * 100, 1) for c in RET + ["lft_long"] if c in sub}
    r["CDI"] = round((np.exp(sub["cdi_ret"].sum()) - 1) * 100, 1)
    r["days"] = len(sub)
    rows[cname] = r
save(pd.DataFrame(rows).T, "tbl_crisis_asset_returns.csv")
print("\n  Asset total returns by crisis (%):")
print(pd.DataFrame(rows).T.to_string())


# ─────────────────────────────────────────────────────────────────────────────
hdr("10. Stressed VaR at a horizon matched to the estimation window")
cov_full = master[RET].dropna().cov()
rows = []
for pname, w in PORTFOLIOS.items():
    row = {"Calm (10d)": round(M.portfolio_var(w, cov_full, 0.99, 10), 1)}
    for cname, (s, e) in CRISES.items():
        sub = master[(master.index >= s) & (master.index <= e)][RET].dropna()
        if len(sub) < 30:
            row[f"{cname} (10d)"] = np.nan          # too few obs for a 5x5 covariance
            continue
        row[f"{cname} (10d)"] = round(M.portfolio_var(w, sub.cov(), 0.99, 10), 1)
    rows.append(pd.Series(row, name=pname))
var_df = pd.DataFrame(rows)
save(var_df, "tbl_stressed_var.csv")
print(var_df.to_string())
print("\n  (Joesley is NaN by design: 11 trading days cannot support a 5x5 covariance.)")

cols = ["ibov", "ntnb"]
cov_sub = cov_full.loc[cols, cols].values
w_arr = np.array([0.60, 0.40])
v0 = np.sqrt(w_arr @ M.shrink_to_equicorr(cov_sub, 0.0) @ w_arr)
v1 = np.sqrt(w_arr @ M.shrink_to_equicorr(cov_sub, 1.0) @ w_arr)
print(f"\n  Equicorrelation stress on 60/40: 10-day VaR "
      f"{-1 * __import__('scipy.stats', fromlist=['norm']).norm.ppf(0.01) * v0 * np.sqrt(10) * 100:.1f}% "
      f"-> {-1 * __import__('scipy.stats', fromlist=['norm']).norm.ppf(0.01) * v1 * np.sqrt(10) * 100:.1f}% "
      f"({(v1 / v0 - 1) * 100:+.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
hdr("11. Rolling portfolio metrics")
W = 252
df_ret = master[RET].dropna()
recs = []
for i in range(W, len(df_ret)):
    win = df_ret.iloc[i - W:i]
    cov = win.cov()
    rec = {"date": df_ret.index[i], "pc1_all": M.pc1_share(win)}
    for pname, w in [("60/40", {"ibov": .6, "ntnb": .4}),
                     ("Diversified", {"ibov": .4, "ntnb": .3, "ltn": .15, "lft": .15})]:
        cols = list(w)
        cs = cov.loc[cols, cols].values
        wa = np.array([w[c] for c in cols])
        rec[f"DR {pname}"]  = M.diversification_ratio(wa, cs)
        rec[f"ENB {pname}"] = M.effective_num_bets(wa, cs)
        rec[f"PC1 {pname}"] = M.pc1_share(win[cols])
    recs.append(rec)
roll = pd.DataFrame(recs).set_index("date")
save(roll.round(4), "tbl_rolling_portfolio_metrics.csv")
print(roll.describe().loc[["mean", "min", "max"]].round(3).to_string())
print(f"\n  NOTE: ENB for the 2-asset 60/40 is bounded above by 2 by construction.")
print(f"  PC1 'all' is computed on all {len(RET)} assets; 'PC1 60/40' on the two held.")

print("\n" + "=" * 78)
print(f"  Wrote {len(tables)} tables to outputs/")
print("=" * 78)
