# Web edition

`index.html` is a self-contained web page presenting the findings of
`docs/04_final_paper.md`. It is published as a Claude artifact and can also be
opened directly in a browser or served as a static file — it has no external
dependencies (no CDN scripts, stylesheets, or font requests).

## Contents

Six hand-built SVG figures plus the paper's tables, all keyed back to the source
section and table numbers:

| Figure | Shows | Source |
|---|---|---|
| 1 | Regime correlations with 95% intervals — the non-negative floor | §4.2, Table 3 |
| 2 | Correlation by return frequency, four bonds | §4.3, Table 5 |
| 3 | Raw vs Forbes-Rigobon adjusted crisis correlation | §6, Table 8 |
| 4 | Empirical lower/upper tail dependence at the 5% threshold | §5.2, Table 7 |
| 5 | Crisis-window return in excess of CDI | §8.2, Table 12 |
| 6 | 99% ten-day Gaussian VaR by portfolio and regime | §8.4, Table 13 |

Every figure carries a "Table" toggle exposing the underlying numbers, so no value
is reachable only through a chart.

## Rebuilding

The page is assembled from `template.html`, which carries `__BODONI__`,
`__ARCHIVO__`, `__PLEX400__` and `__PLEX500__` placeholders in its `@font-face`
rules. `build.py` base64-encodes the woff2 files in `fonts/` into those slots and
writes `index.html`:

```bash
cd site && python3 build.py
```

Edit `template.html`, never `index.html` — the latter is generated.

## Data

Figures are transcribed from the tables in `docs/04_final_paper.md`, which
`scripts/run_analysis.py` regenerates from public sources. The page does not run
the pipeline; if a table in the paper changes, update the corresponding array at
the top of the script block in `template.html` and rebuild.

## Fonts

Bodoni Moda (display), Archivo (body) and IBM Plex Mono (data), latin subsets,
all under the SIL Open Font License and inlined as data URIs so the page renders
identically offline and under a strict content security policy.
