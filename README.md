# CapstoneBO (HEBO-inspired Weekly BO)

This folder is the active weekly BO pipeline with HEBO-inspired upgrades.
Legacy baseline materials live under `Capstone/Legacy Files/BOTorch/*`.

## Documentation

- [BBO Capstone Dataset Datasheet](docs/datasheet_bbo_capstone_dataset.md)
- [BBO Optimisation Approach Model Card](docs/model_card_bbo_optimisation_approach.md)

## What this adds

- Output power transform before GP fitting:
  - Box-Cox when all targets are strictly positive (or strictly negative after sign flip)
  - Yeo-Johnson otherwise
- Input warping on `[0,1)` via Kumaraswamy CDF:
  - uses BoTorch `Warp` when available
  - fallback learnable Kumaraswamy warp module otherwise
- Kernel mode switch:
  - `baseline`
  - `lin_matern32_add` = Linear + Matérn(ν=1.5) with ARD (wrapped by `ScaleKernel`)
- MACE-style acquisition aggregation:
  - score candidate pool with EI / PI / UCB(beta sweep)
  - compute Pareto front on (−EI, −PI, −UCB)
  - select a single weekly submission via sigma-quantile gate then max posterior mean
- Stable EI path:
  - prefers `LogExpectedImprovement`
  - repairs EI values if numeric issues occur
- Portal rounding guard:
  - 6dp floor quantization
  - duplicate rejection after rounding against observed points
- Optional boundary-hugging control for higher-dimensional functions:
  - penalizes near-edge points in tie-break scoring (`mu_t_tiebreak = mu_t - penalty`)
  - defaults: `--boundary_penalty_weight 0.10 --boundary_penalty_min_dim 4 --boundary_margin 0.02`
  - disable with `--boundary_penalty_weight 0.0`

## Run weekly report

```bash
python3 -m Capstone.CapstoneBO.weekly_pack \
  --report_date 2026-02-24 \
  --seed 0
```

or

```bash
python3 Capstone/CapstoneBO/weekly_pack.py --report_date 2026-02-24 --seed 0
```

Selection defaults to robust consensus across seeds `0 1 2 3 4`.
You can switch modes:

```bash
# robust consensus (default)
python3 -m Capstone.CapstoneBO.weekly_pack \
  --selection_mode robust_consensus \
  --consensus_seeds 0 1 2 3 4 \
  --consensus_min_successes 3

# single seed compatibility mode
python3 -m Capstone.CapstoneBO.weekly_pack \
  --selection_mode single_seed \
  --seed 0
```

Outputs:

`Capstone/CapstoneBO/reports/YYYY-MM-DD/`

- `candidate_summary.csv`
- `recommendations.csv`
- `portal_strings.csv`
- `candidate_shortlist.csv`
- `candidate_pivot.csv`
- `seed_suggestions.csv`
- `stability_summary.csv`
- `run_health.json`
- `weekly_dashboard.html` (unless `--no_dashboard`)
- `function_k/model_diagnostics.json`

## Quick propose-only mode

```bash
python3 -m Capstone.CapstoneBO.propose_next --seed 0
```

Prints one portal string per function.

## High-Dimensional Visualization Summaries (Functions 3-8)

Generate one 2x3 summary PNG per function with:
- PCA geometry view,
- feature loading/correlation diagnostics,
- t-SNE neighborhood scatter,
- GP mean 2D slice on top-ranked features,
- 1D partial dependence curves,
- target distribution and run stats.

```bash
python3 -m Capstone.CapstoneBO.plot_highdim_function_summaries \
  --functions function_3 function_4 function_5 function_6 function_7 function_8 \
  --seed 0
```

or

```bash
python3 Capstone/CapstoneBO/plot_highdim_function_summaries.py --seed 0
```

Outputs are written to:

`Capstone/CapstoneBO/images/function_k_summary.png`

Defaults:
- input data path: `Capstone/CapstoneBO/data`
- output path: `Capstone/CapstoneBO/images`
- grid size: `120`
- top features for 1D PD curves: `2`

## How To Interpret The Summary PNGs

Each `function_k_summary.png` has six panels. Read them together:

1. `PCA Scatter (PC1 vs PC2)`:
   - Points close together have similar input settings in reduced space.
   - Strong colour gradients suggest target structure aligned with major variation directions.
   - Mixed colours with no visible pattern suggest weak global low-dimensional structure.

2. `PCA Loading Strength`:
   - Larger bars mean that feature contributes more to the first two PCs.
   - Use this to identify which inputs drive most geometric variation in the observed data.

3. `t-SNE Scatter`:
   - This is a nonlinear 2D projection of the observed inputs that tries to preserve local neighborhoods from the original high-dimensional space.
   - Points close together in this panel tend to represent similar input settings, even though the axes themselves do not map back to original features.
   - Clusters of similar colours suggest neighborhoods with consistently similar outcomes.
   - In the default colour scale, a yellow cluster surrounded by purple-to-green points usually marks a promising local neighborhood to exploit, though not necessarily the globally best region.
   - Isolated points can indicate unusual or underexplored samples, and mixed colours within a neighborhood suggest a less smooth local response.

4. `GP Mean Slice (2D)`:
   - This is model-estimated response over the two selected features, with other features fixed at medians.
   - High-value regions are candidate zones for exploitation.
   - Sharp transitions or narrow ridges indicate sensitive regions where small input changes matter.

5. `1D Partial Dependence (GP Mean)`:
   - Shows model trend for each top feature independently (others fixed at medians).
   - Steeper curves imply stronger local effect on predicted outcome.
   - Flat curves suggest weaker single-feature influence or interaction-dominated behavior.

6. `Target Distribution + Stats`:
   - Histogram shows spread/skew of observed `y0`.
   - Text block gives `n`, `d`, min/max/median, selected top features, and `|corr(x, y0)|` ranking.
   - Use this to contextualise whether the model is learning from broad variation or a narrow target band.

Practical BO read:
- Prefer regions where panel 4 is high and panel 5 is not excessively unstable unless you want exploration.
- If panel 3 shows isolated points or sparse neighborhoods, treat that as a cue to inspect uncertainty and novelty-driven exploration.
- If panel 3 has mixed colours within local neighborhoods but panel 4 is structured, the model may be extracting signal that is not obvious from the raw neighborhood view alone.
- If all panels look weakly structured, rely more on uncertainty-aware acquisitions (EI/PI/UCB) than pure mean chasing.
- Use `best_exploration` to inspect potential blind spots: it surfaces points with high uncertainty and high novelty relative to observed data.

## Exploration Blind-Spot Signal

To reduce the chance of missing under-sampled regions, reports now include:
- `exploration_score = sigma_t * novelty_ratio`
- `novelty_ratio = nearest_distance_to_observed / median_nearest_distance_observed`

Where it appears:
- `candidate_summary.csv` column: `exploration_score`
- `recommendations.csv` row type: `best_exploration`
- `weekly_dashboard.html`: includes the same `best_exploration` recommendation row

Interpretation:
- High `sigma_t` alone can still point to already-near-observed boundary areas.
- `exploration_score` boosts candidates that are both uncertain and geometrically far from known data.
- Final submission still follows the Pareto + sigma-gate + mean tie-break path; `best_exploration` is an explicit alternative probe recommendation.

## Notes

- Acquisitions are computed in transformed-target space (consistent with model training).
- `mu_raw_est` in reports is an inverse-transformed approximation of posterior mean.
- `shadow_oldstyle` recommendation is exported for side-by-side comparison with legacy-style `mu + 0.5*sigma` selection.
- `best_exploration` recommendation is exported using `exploration_score = sigma_t * novelty_ratio` to surface high-uncertainty, under-sampled regions.
- Model fitting now uses fallback attempts (warp/transform/kernel/jitter variants) and logs the selected attempt in diagnostics.
- Use `stability_summary.csv` and `run_health.json` to decide trust:
  - high `consensus_frequency_ratio` and low failure counts -> higher trust,
  - low seed success or frequent fallback -> treat as exploratory / lower confidence.
