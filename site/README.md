# Web edition

`index.html` is a self-contained executive one-pager presenting the headline findings of
`docs/04_final_paper.md`. It is published as a Claude artifact and can also be opened
directly in a browser or served as a static file — it has no external dependencies (no
CDN scripts, stylesheets, or font requests).

## Contents

Four blocks, no section-by-section walkthrough:

| Block | Carries |
|---|---|
| Masthead | Title, thesis, byline, sample window |
| Hero figure | Regime correlations with 95% intervals — the non-negative floor (§4.2, Table 3) |
| Scorecard rail | Daily ρ, monthly ρ, regimes distinguishable, Ibovespa Sharpe vs CDI |
| Conclusion | The allocation implication, with the crisis-cushion and tail-dependence figures inline |
| Qualifications | Bootstrap significance, Forbes-Rigobon, Gaussian VaR, the unreproduced §9 |

The hero figure carries a **Table** toggle exposing all seven regime rows with intervals,
bootstrap standard errors and p-values, so the chart is not the only route to a number.
Hovering a row gives n, the interval and the bootstrap comparison.

### The qualifications block is content, not decoration

The scorecards quote point estimates. The qualifications are what keep quoting them
honest — above all that **no regime is statistically distinguishable from the calmest
one**, which makes the visible upward drift across regimes suggestive rather than
established. Don't drop that block to save vertical space.

For everything else — the horizon table, the Forbes-Rigobon decomposition, the copula and
DCC estimates, portfolio metrics, stressed VaR, the nine findings and the full limitations
— read the paper. A long-form edition covering all six figures is in git history at commit
`4af882c` if it is ever wanted back.

## Rebuilding

The page is assembled from `template.html`, which carries `__BODONI__`, `__ARCHIVO__`,
`__PLEX400__` and `__PLEX500__` placeholders in its `@font-face` rules. `build.py`
base64-encodes the woff2 files in `fonts/` into those slots and writes `index.html`:

```bash
cd site && python3 build.py
```

Edit `template.html`, never `index.html` — the latter is generated and gets overwritten.
`build.py` fails loudly if a placeholder is missing rather than emitting a page whose
fonts silently fall back.

## Layout notes

- **Height.** ~1190px at 1280px wide: one screen on a tall display, one short scroll on a
  900px-tall laptop. Getting below that would mean cutting either the conclusion's
  findings or the qualifications, so it stays as is.
- **The hero row** (chart + scorecard rail) collapses to full width below 1120px, because
  the chart needs ~800px to keep its axis labels legible. Below ~770px the chart holds a
  700px floor and its plate scrolls sideways instead, with a visible hint.
- **Teal is the only data hue.** With a single chart there is no second series, so the
  page carries no categorical partner colour and no sequential ramp.

## Data

Figures are transcribed from the tables in `docs/04_final_paper.md`, which
`scripts/run_analysis.py` regenerates from public sources. The page does not run the
pipeline; if a table in the paper changes, update the `REGIMES` array at the top of the
script block in `template.html`, along with the figures quoted inline in the scorecards,
conclusion and qualifications, and rebuild.

## Fonts

Bodoni Moda (display), Archivo (body) and IBM Plex Mono (data), latin subsets, all under
the SIL Open Font License and inlined as data URIs so the page renders identically
offline and under a strict content security policy.
