# Web edition

`index.html` is a self-contained results dashboard for `docs/04_final_paper.md`. It is
published as a Claude artifact and can also be opened directly in a browser or served as a
static file — it has no external dependencies (no CDN scripts, stylesheets, or font
requests).

## Contents

| Block | Carries |
|---|---|
| Masthead | Title, byline, sample window |
| Why block | The sovereign-credit channel — the mechanism behind the floor (§2.1) |
| Scorecards | Daily ρ, monthly ρ, the 60-month floor, tail co-crash, crises cushioned, Ibovespa Sharpe |
| Six panels | One chart each, two across |
| Conclusion | The allocation implication, including the negative equity premium |
| Qualifications | What the estimates do not establish |

The six panels, in reading order:

| Fig | Shows | Paper |
|---|---|---|
| 1 | Correlation by macro regime with 95% intervals — the floor | §4.2 |
| 2 | Minimum 60-month rolling correlation, Brazil vs the G4 | §9.4 |
| 3 | Crisis correlations raw vs Forbes-Rigobon adjusted | §6 |
| 4 | Correlation across daily → quarterly horizons, four instruments | §4.3 |
| 5 | Excess return over CDI by crisis episode, NTN-B and 60/40 | §8.2 |
| 6 | Lower vs upper tail dependence against the independence benchmark | §5.2 |

There are **no tables**. Every number a panel carries is either printed on the chart or in
its hover tooltip, which gives n, intervals, bootstrap comparisons and copula fits.

### Two editorial rules

- **Captions answer *why*, not *how*.** A caption says what the panel changes about an
  allocation decision. Method — return construction, Fisher-z intervals, bootstrap design —
  lives in the paper and does not belong on a dashboard.
- **The qualifications block is content, not decoration.** The scorecards quote point
  estimates; these lines are what keep quoting them honest, above all that no regime is
  statistically distinguishable from the calmest. Don't drop it to save vertical space.

For the horizon table, the DCC estimates, stressed VaR, CoVaR, the twelve findings and the
full limitations, read the paper. A long-form editorial edition is in git history at commit
`4af882c`, and the executive one-pager it replaced at `8ec4a61`.

## Rebuilding

The page is assembled from `template.html`, which carries `__BODONI__`, `__ARCHIVO__`,
`__PLEX400__` and `__PLEX500__` placeholders in its `@font-face` rules. `build.py`
base64-encodes the woff2 files in `fonts/` into those slots and writes `index.html`:

```bash
cd site && python3 build.py
```

Edit `template.html`, never `index.html` — the latter is generated and gets overwritten.
`build.py` fails loudly if a placeholder is missing rather than emitting a page whose fonts
silently fall back.

## Layout notes

- **One chart width.** Every figure is drawn on a **530-unit viewBox**, because a panel body
  measures 532px inside the 1280px container. That makes charts render 1:1, so an 11px axis
  label is 11px on screen. `.plate-body` therefore carries **no horizontal padding** —
  `clientWidth` includes padding, and 2×1.1rem of it would scale every label down to 10.3px.
  If you change the container width, the grid, or that padding, re-measure and re-tune the
  viewBox rather than letting the charts scale.
- **Breakpoints.** Panels go one-across below 1120px; the charts hold a 470px floor and their
  bodies scroll sideways below that, with a visible hint.
- **Three data colours.** `--d1` teal is the estimate being argued from, `--d2` ochre is
  whatever it is contrasted against (raw vs adjusted, upper vs lower tail), `--neg` is loss.
  The trio passes the categorical checks — OKLCH lightness band, chroma floor, protan/deutan
  separation, normal-vision ΔE and 3:1 contrast — in **both** themes, all pairs. Re-run the
  validator if you touch them; the dark `--neg` is `#e0687f` specifically because the obvious
  brick red sits ΔE 4.8 from the ochre under deuteranopia.

## Data

Figures are transcribed from the tables in `docs/04_final_paper.md`, which
`scripts/run_analysis.py` regenerates from public sources. The page does not run the
pipeline; if a table in the paper changes, update the corresponding array at the top of the
script block in `template.html` (`REGIMES`, `FLOOR`, `HORIZON`, `CRISES`, `EXCESS`, `TAIL`),
along with the figures quoted in the scorecards, conclusion and qualifications, and rebuild.

## Fonts

Bodoni Moda (display), Archivo (body) and IBM Plex Mono (data), latin subsets, all under the
SIL Open Font License and inlined as data URIs so the page renders identically offline and
under a strict content security policy.
