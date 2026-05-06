# CapstoneBO+ Explainer

This document explains what the `CapstoneBO+` code does.

## Purpose

`CapstoneBO+` is a weekly Bayesian Optimization (BO) pipeline that:
- reads historical observations from `function_*.csv` files
- fits one GP model per function
- scores a mixed candidate pool with multiple acquisition functions
- chooses one weekly submission candidate per function with a small portfolio of selection strategies
- writes CSV/JSON/HTML reports and portal-ready strings for submission

It starts from the `CapstoneBO` workflow and adds:
- output power transforms (Box-Cox / Yeo-Johnson)
- input warping
- configurable kernel modes
- mixed global / trust-region / elite candidate generation
- Multi-objective Acquisition function ensemble style selection plus TS-style scoring
- portfolio switching between exploitative and exploratory sub-strategies
- explicit stagnation detection
- robust fitting fallback attempts for unstable GP optimisation
- consensus selection across multiple seeds for final portal suggestions

## Differences vs `CapstoneBO`

An original `CapstoneBO` codebase was primarily a global strategy:
- draw one large Sobol pool over the full search box,
- score it with EI / PI / UCB,
- Pareto-filter candidates,
- apply a sigma gate,
- tie-break by posterior mean.

`CapstoneBO+` keeps that baseline logic, but added three major upgrades.

1. Mixed candidate generation instead of only global Sobol:
- `global` candidates still cover the full domain
- `tr` candidates are sampled inside a reconstructed trust region around the current strong incumbent
- `elite` candidates are sampled inside a box built from the best observed region so far
2. Portfolio recommendations instead of one dominant selector:
- `primary_pareto_sigma` preserves the original selection philosophy
- `trust_region_ts` uses a trust-region-biased Thompson-style score
- `global_explore` explicitly prefers uncertain and novel points
- `shadow_oldstyle` is still exported for continuity against the older exploit-plus-uncertainty rule

3. Explicit stagnation handling:
- `CapstoneBO` had robust fitting fallbacks, but there was still a chance that the optimisation policy could get stuck.
- `CapstoneBO+` computes a `stagnation_flag` from recent best-so-far progress using a scale-aware score based on the observed response spread.
- When progress stalls, the default stagnant policy is a hybrid switch. We compare the best trust-region Thompson-style candidate against the best non-trust-region exploration candidate and choose the more convincing regime.

## Main Entrypoints

- `weekly_pack.py`
  - CLI + callable `run_weekly(cfg, write_outputs=True)`.
  - Default mode is multi-seed consensus (`selection_mode=robust_consensus`).
- `propose_next.py`
  - Lightweight CLI wrapper around `run_weekly(..., write_outputs=False)`.
  - Prints one portal string per function.

## High-Level Data Flow

1. Build config (`RebuildConfig`) from CLI flags.
2. Discover function CSVs in `data_dir`.
3. For each `function_k.csv` across selected seeds:
   - load and sanitize data,
   - fit transformed GP (`fit_warped_gp`) with fallback attempts,
   - generate Sobol candidate pool,
   - compute acquisition scores (EI / logEI / PI / UCB over beta sweep),
   - deduplicate candidates after portal rounding and against observed points,
   - select submission by Pareto-front strategy,
   - assemble candidate/recommendation/portal rows,
   - collect model diagnostics.
4. Aggregate per-seed submissions into one final portal string per function.
5. Write outputs (`candidate_summary.csv`, `recommendations.csv`, `portal_strings.csv`, `seed_suggestions.csv`, `stability_summary.csv`, `run_health.json`, diagnostics JSONs, optional dashboard).

## Core Modules

## `config.py`

- Defines immutable `RebuildConfig` dataclass for all run controls:
  - model settings (`kernel_mode`, transforms),
  - candidate settings (`n_sobol`, `topk_per_acq`, `ucb_betas`),
  - portal formatting (`decimals`, bounds),
  - reliability/selection settings (`selection_mode`, `consensus_seeds`, `consensus_min_successes`, `fit_retry_max_attempts`).
- `sigma_quantile_schedule(n_points)` controls exploration gate strictness:
  - `<20` points: 0.50 quantile,
  - `<35` points: 0.35,
  - else: 0.20.

Interpretation: with more historical data, it allows lower-sigma candidates into the final gate.

## `data_io.py`

- `used_x_cols(df)` finds populated feature columns `x0, x1, ...`.
- `load_function_dataset(csv_path, dtype)`:
  - validates `y0` exists,
  - drops rows missing used x-columns or `y0`,
  - returns `FunctionDataset` containing `X`, `y`, metadata.
- `list_function_csvs(data_dir)` finds `function_*.csv`.
- `make_bounds(...)` returns simple tensor bounds helper.

## `transforms/output_power.py`

`OutputPowerTransform` wraps scikit-learn `PowerTransformer` for target normalization:
- uses Box-Cox if all targets are strictly positive,
- uses Box-Cox on sign-flipped values if all strictly negative,
- uses Yeo-Johnson otherwise,
- identity fallback for tiny data (`n < 2`),
- supports `transform()` and `inverse_transform()`.

This improves GP fit when targets are skewed/heavy-tailed.

## `models/kernels.py`

`make_covar_module(kernel_mode, d)`:
- `baseline`: `ScaleKernel(MaternKernel(nu=2.5, ARD))`
- `lin_matern32_add`: `ScaleKernel(Linear + Matern(nu=1.5), ARD)`

Allows you to switch between different kernels that get used by `fit_warped_gp`.

## `models/warped_gp.py`

Main model fitting logic.

- Tries to use BoTorch input `Warp` if available.
- If not, uses `KumaraswamyWarpFallback`:
  - learnable per-dimension parameters `a`, `b`,
  - transform `w(x) = 1 - (1 - x^a)^b`, with softplus constraints.
- `fit_warped_gp(X, y, cfg)`:
  - runs a fallback chain (primary, no-warp/no-transform, baseline kernel variants, higher-jitter variants),
  - accepts the first successful fit,
  - records the selected attempt and failed attempts.
- Returns `FittedWarpedGP` with model + transform objects + diagnostics helper.

Diagnostics include estimated noise, lengthscale summaries, warp parameters, transform metadata.

## `acquisition/scores.py`

Computes candidate scores from posterior and acquisition functions.

- `compute_acq_scores(model, y_t, X_pool, ucb_betas)` returns:
  - posterior `mu_t`, `sigma_t`,
  - `ei`, `log_ei`, `pi`,
  - per-beta UCB arrays,
  - `ucb_max` and winning beta per candidate.

Numerical stability details:
- Prefers `LogExpectedImprovement` when available.
- Derives or repairs EI from logEI path when EI has bad values.

## `candidates/pool.py`

- Draws Sobol candidates: `n_sobol x d`.
- Scores all points with `compute_acq_scores`.
- Builds a `priority_indices` set by union of top-k points from:
  - logEI, EI, PI, max variance, and each UCB(beta).

## `acquisition/mace_pareto.py`

Pareto utilities (minimization space):
- `pareto_front_indices(objectives_min)`,
- `pareto_ranks(objectives_min)`.

Used to perform non-dominated filtering on multi-acquisition objectives.

## `candidates/selectors.py`

`select_submission_from_pareto(...)` chooses one candidate from scored pool:

1. Build minimization objectives from `(-EI, -PI, -UCB)`.
2. Compute Pareto front + ranks.
3. Apply sigma gate on Pareto set using `sigma_quantile_schedule(n_points)`.
4. Tie-break in order:
   - max tie-break mean (defaults to `mu_t`; can be adjusted, e.g. boundary-penalized `mu_t_tiebreak`),
   - then max `EI`,
   - then max `PI`,
   - then smallest index (determinism).

Returns `SelectionResult` including selected index, pareto indices/ranks, sigma threshold, and gated set.

## `rounding.py`

Portal-safe formatting and deduplication.

- Quantisation is floor-based to fixed decimals, clipped to `[lower, upper]`.
- `format_portal_string` outputs hyphen-separated portal format.
- `dedup_after_rounding(candidates, observed, scores, ...)`:
  - removes candidates that collide with observed points after quantisation,
  - removes internal candidate collisions after quantisation,
  - keeps highest-score candidate for each quantised key.

This prevents submitting duplicates that only differ before rounding.

## `weekly_pack.py`

Orchestrates the complete weekly workflow.

Important helper functions:
- `_torch_dtype` maps config string to torch dtype.
- `_nn_stats`, `_nearest_distance`, `_boundary_prox` compute novelty/boundary metrics.
- `exploration_score = sigma_t * novelty_ratio` is computed to surface high-uncertainty, under-sampled candidates.
- `_make_random_nonduplicate` fallback candidate if dedup leaves nothing.
- `_apply_function_overrides` per-function config patching.

`run_weekly(cfg, write_outputs)` pipeline:
- run each function for one or multiple seeds,
- keep per-seed suggestion rows and failure details,
- choose final portal string by frequency consensus (tie-break: mean submission score, then mean `mu_t`, then lexical),
- fallback when successful seeds are below threshold,
- write representative candidate/recommendation tables plus health/stability diagnostics.

Recommendation anchors include:
- `chosen_submission`, `best_mu`, `best_sigma`, `best_logEI`, `best_PI`, `best_UCB`, `shadow_oldstyle`
- `best_exploration` (max `exploration_score`) for explicit blind-spot probing

CLI `main()` just parses args, builds config, runs once, writes report directory, prints key files.

## `reporting/export_csv.py`

Writes report artifacts:
- `candidate_summary.csv`
- `recommendations.csv`
- `portal_strings.csv`
- `seed_suggestions.csv`
- `stability_summary.csv`
- `run_health.json`
- compatibility helpers:
  - `candidate_shortlist.csv`
  - `candidate_pivot.csv`
- per-function `model_diagnostics.json`
- optional `weekly_dashboard.html`

## `reporting/dashboard.py`

Generates a simple static HTML report from candidates + recommendations tables.

## `propose_next.py`

Convenience command to print one portal string per function without writing files:
- builds a `RebuildConfig`,
- calls `run_weekly(..., write_outputs=False)`,
- prints sorted `function -> portal_string`.

## Package `__init__.py` Files

These re-export key functions/classes for cleaner imports:
- top level: `RebuildConfig`
- acquisition: `compute_acq_scores`, `pareto_front_indices`
- candidates: `build_candidate_pool`, `select_submission_from_pareto`
- models: `FittedWarpedGP`, `fit_warped_gp`
- reporting: `write_report_csvs`
- transforms: `OutputPowerTransform`

## Output Files You Should Read First

- `portal_strings.csv`: what you actually submit.
- `recommendations.csv`: why selected candidates were chosen vs anchor strategies.
- `candidate_summary.csv`: full scored pool and diagnostics flags.
- `seed_suggestions.csv`: per-seed chosen submissions (including failed seed/function attempts).
- `stability_summary.csv`: per-function agreement and fallback metrics.
- `run_health.json`: global warning/failure counters for trust checks.
- `function_k/model_diagnostics.json`: model-level diagnostics (noise, warp, transform, kernel metadata).

## Practical Interpretation

- Selection is a multi-objective strategy:
  - build Pareto-efficient set across EI/PI/UCB
  - require enough uncertainty via sigma gate
  - then choose best posterior mean among gated Pareto points.
- `best_exploration` is intentionally separate from the final submission rule:
  - it identifies candidates with both high uncertainty and high novelty,
  - useful when you suspect coverage blind spots (especially in higher dimensions where visual inspection is hard).
- Rounding/dedup logic is critical because portal precision constraints can collapse distinct float candidates into the same submitted point.

## Reading the High-Dimensional Summary Plots

For functions `3` through `8`, the high-dimensional summary figure is a compact diagnostic view that combines embeddings, GP summaries, and target statistics.

### PCA panel

- The PCA scatter projects the observed input points down to two linear components (`PC1`, `PC2`).
- Points that are close together have similar coordinates in the original input space after a linear projection.
- Point colour still represents `y0`, so colour structure tells you whether good or bad outcomes align with broad linear trends.
- The percentages in the title show how much input variation is explained by the first two principal components.

How to interpret PCA:
- if similarly colored points separate along PC1 or PC2, the function may have some broad, structured trend in the observed region
- if colours are heavily mixed, either the relationship is more complex/nonlinear or the first two PCs are not capturing the important directions
- if the explained-variance percentages are low, the 2D PCA view is only showing a small slice of the geometry

What PCA is good for:
- broad global structure,
- rough separation of regions,
- seeing whether the observed data cloud is stretched along a few dominant directions.

### PCA loading panel

- The loading bar chart shows which original features contribute most strongly to the first two principal components.
- Larger bars mean that feature has more influence on the PCA view.

### t-SNE panel

- The t-SNE scatter also projects observed input points into 2D, but unlike PCA it is nonlinear.
- Its main purpose is to preserve local neighborhoods: points that are close in the t-SNE plot tend to be similar in the original high-dimensional input space.
- Point colour again represents `y0`.

How to interpret t-SNE:
- clusters of similar colour suggest regions of input space with consistently similar outcomes;
- in the default colour scale, a yellow cluster surrounded by purple-to-green points usually marks a promising local neighborhood to exploit, though not necessarily the globally best region;
- isolated points may indicate unusual or underexplored samples;
- a smooth colour transition across nearby points can suggest a locally smooth response surface.

### GP mean slice and 1D partial dependence

- The GP mean slice is a 2D model-based view using two selected features while holding the others near their median values.
- The 1D partial dependence curves show how the fitted GP mean changes as one feature moves and the others stay fixed.

These panels are useful for intuition, but they are conditional model summaries, not direct observations of the true function everywhere.

### Best practice for reading the whole figure

- Use PCA to understand broad linear structure.
- Use t-SNE to understand local clustering and neighborhood structure.
- Use the GP slice and 1D dependence panels to form hypotheses about feature effects.
- Use the target-statistics panel to sanity-check sample size, target spread, and which features appear most associated with `y0`.