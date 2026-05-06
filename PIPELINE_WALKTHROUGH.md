# CapstoneBO+ Pipeline Walkthrough

This document explains the `CapstoneBO+` pipeline in the order it runs.

It is written for someone who:
- knows basic Python syntax,
- can read functions and classes,
- but is still getting comfortable reading a medium-sized codebase.

The goal is not just to say "what files exist." The goal is to explain:
- what happens first,
- what object gets created at each stage,
- why that stage exists,
- and how the code moves from raw CSV data to one final portal-ready submission string.

## 1. The Big Picture

Each weekly run of `CapstoneBO+` does roughly this:

1. Build a configuration object that says how the run should behave.
2. Find the `function_*.csv` files in the data folder.
3. For each function, and often for several random seeds:
   - load the observed data,
   - transform the target values,
   - fit a Gaussian process model,
   - build a pool of candidate points,
   - score those candidates with several acquisition functions,
   - remove candidates that would become duplicates after portal rounding,
   - choose one candidate using a portfolio of selection rules.
4. If multiple seeds were used, compare the seed-level choices and pick a consensus portal string.
5. Write CSV, JSON, and HTML outputs so you can inspect what happened.

If you only remember one sentence, remember this:

`CapstoneBO+` is a weekly "read data -> fit model -> score candidate points -> choose one safe submission -> write reports" pipeline.

## 2. The Main Entrypoints

There are two top-level scripts most people care about:

- `CapstoneBO+/propose_next.py`
- `CapstoneBO+/weekly_pack.py`

### `propose_next.py`: the simple wrapper

This file is the lighter entrypoint. Its job is:
- build a config,
- call `run_weekly(...)` from `weekly_pack.py`,
- skip writing report files,
- print one portal string per function.

So when you run:

```bash
python3 CapstoneBO+/propose_next.py --seed 0
```

you are still using the main pipeline. You are just using it in a "give me the next suggestions quickly" mode.

Important beginner idea:

`propose_next.py` does not contain the real optimization logic itself. It is mostly a wrapper around the real engine in `weekly_pack.py`.

### `weekly_pack.py`: the real orchestrator

This is the most important file in the repository.

Its central callable is:

```python
run_weekly(cfg, write_outputs=True)
```

That function is the conductor of the whole system. It decides:
- which functions to process,
- which seeds to run,
- when to call the model-fitting code,
- when to call candidate generation,
- when to call selection,
- and when to write reports.

If you are trying to understand the pipeline, `weekly_pack.py` is the best place to start.

## 3. Stage 1: Build the Configuration

The config lives in:

- `CapstoneBO+/config.py`

The main class is:

```python
@dataclass(frozen=True)
class RebuildConfig:
    ...
```

### What this means

This is a Python `dataclass`. In plain English, that means:
- it is a class mostly used to store named pieces of data,
- Python automatically gives it an initializer like `RebuildConfig(...)`,
- because it is `frozen=True`, the object is meant to be treated as immutable after creation.

So `RebuildConfig` is the pipeline's settings bundle.

### Why the config matters

Instead of scattering constants all over the code, the pipeline puts most decisions in one object:
- where to find the data,
- where to write reports,
- which kernel to use,
- whether to use input warping,
- how many Sobol candidates to generate,
- which UCB betas to test,
- whether to run a single seed or a multi-seed consensus run.

That means functions can stay cleaner. Instead of taking twenty tiny arguments, many of them just take `cfg`.

### Examples of settings in `RebuildConfig`

- `data_dir`: where `function_*.csv` files live
- `reports_dir`: where output folders should be written
- `seed`: the main seed for reproducibility
- `kernel_mode`: which GP kernel family to use
- `output_transform_mode`: whether to use the power transform or copula transform
- `selection_mode`: whether to use one seed or consensus across several seeds
- `n_sobol_global`, `n_sobol_tr`, `n_sobol_elite`: how many candidates to sample from each source
- `decimals`: how many decimals to keep for portal formatting

### The sigma gate schedule

The same file also defines:

```python
def sigma_quantile_schedule(n_points: int) -> float:
    ...
```

This small helper controls how strict the uncertainty filter should be during selection.

The idea is:
- with very little data, keep more uncertainty in play,
- with more data, you can afford to be a bit more selective.

This function gets used later when the pipeline narrows the Pareto set to a final candidate.

## 4. Stage 2: Discover and Load the Data

The loading helpers live in:

- `CapstoneBO+/data_io.py`

### `list_function_csvs(...)`

This function scans the data directory for files named like:

- `function_1.csv`
- `function_2.csv`
- ...

If it finds none, it raises an error. That is a good example of a "fail fast" helper: it stops the run early instead of letting the rest of the pipeline operate on missing inputs.

### `load_function_dataset(...)`

This function takes one CSV path and turns it into a structured object:

```python
@dataclass
class FunctionDataset:
    function_id: str
    csv_path: Path
    df_raw: pd.DataFrame
    x_cols: list[str]
    X: torch.Tensor
    y: torch.Tensor
```

This object is useful because it keeps all the related pieces together:
- the original filtered pandas DataFrame,
- the names of the feature columns,
- the input matrix `X`,
- and the target vector `y`.

### What loading actually does

When `load_function_dataset(...)` runs, it:

1. Reads the CSV with pandas.
2. Finds the populated feature columns such as `x0`, `x1`, `x2`, and so on.
3. Checks that `y0` exists.
4. Drops rows that are missing any used `x` column or `y0`.
5. Converts the resulting data into PyTorch tensors.

### Why PyTorch tensors?

This pipeline uses BoTorch and GPyTorch for Gaussian process modeling. Those libraries expect tensors, not plain NumPy arrays.

So a very common pattern in this repository is:
- pandas for reading and cleaning tables,
- NumPy for lightweight numeric work,
- PyTorch for GP model inputs and outputs.

You will see the code move between those three representations a lot.

## 5. Stage 3: Decide Which Seeds to Run

Back in `weekly_pack.py`, `run_weekly(...)` decides whether to run:
- one seed, or
- several seeds and then do consensus.

### Single-seed mode

If:

```python
cfg.selection_mode == "single_seed"
```

then the pipeline does one run and uses that result directly.

### Robust-consensus mode

If:

```python
cfg.selection_mode == "robust_consensus"
```

then the pipeline runs multiple seeds, usually:

```python
(0, 1, 2, 3, 4)
```

and compares the final portal strings across seeds.

### Why use multiple seeds?

Some parts of Bayesian optimization are randomized:
- scrambled Sobol sampling,
- Thompson-style sampling,
- sometimes model fitting behavior,
- and candidate ordering effects.

Running several seeds gives you a simple robustness check:

"Do several independent versions of this pipeline agree on roughly the same answer?"

If they do, confidence is higher. If they disagree strongly, the run is less stable.

## 6. Stage 4: Run One Function for One Seed

The main inner worker is:

```python
_run_one_function(...)
```

This function handles one combination of:
- one function dataset,
- one configuration,
- one per-function seed.

This is the core of the pipeline.

Its output is another dataclass:

```python
@dataclass
class FunctionSeedResult:
    success: bool
    function: str
    seed: int
    ...
```

This object is the "everything that happened for one function under one seed" bundle. It stores:
- whether the run succeeded,
- candidate rows for the report,
- recommendation rows,
- portal output,
- diagnostics,
- and some summary scores.

This is a recurring design style in the repository:
- compute several related values,
- put them into one named dataclass,
- return that object instead of returning a long tuple.

That makes the code easier to read later because `result.submission_portal_string` is clearer than `result[8]`.

## 7. Stage 5: Detect Stagnation and Reconstruct the Trust Region

Before candidate generation, the pipeline computes two useful pieces of state:

- `stagnation_flag`
- `tr_state` (trust-region state)

### `stagnation_flag`

This is computed in:

- `_stagnation_flag(...)` inside `weekly_pack.py`

It looks at recent best-so-far progress in the observed `y` values.

The broad idea is:
- if the best value has not improved over a recent window,
- treat the search as stagnating.
- the improvement test is scale-aware, using a robust spread estimate from the observed `y` values rather than one absolute threshold for every function.

Why this matters:

The pipeline can switch selection behavior when it thinks progress has stalled. In other words, stagnation is not just a statistic for the report; it can influence strategy.

### `reconstruct_tr_state(...)`

This lives in:

- `CapstoneBO+/candidates/trust_region.py`

It reconstructs a trust-region state from the historical data.

The returned object is:

```python
@dataclass
class TrustRegionState:
    center: np.ndarray
    length: float
    success_count: int
    failure_count: int
    n_restarts: int
```

### What is a trust region here?

A trust region is a local box around a promising point. The intuition is:

"Do not always search everywhere. Sometimes spend extra attention near the current best area."

### How reconstruction works

This code walks through the historical observations in order and simulates a simple success/failure rule:
- if a new observation improves enough, count it as a success,
- if not, count it as a failure,
- repeated successes can expand the local box,
- repeated failures can shrink it,
- if the box becomes too small, restart it.

This is not doing a full live TuRBO optimization loop inside the history. It is rebuilding a plausible trust-region state from the data you already have so the current weekly run can sample around that area.

## 8. Stage 6: Transform the Target Before Fitting the GP

The transform code lives in:

- `CapstoneBO+/transforms/output_power.py`
- `CapstoneBO+/transforms/output_copula.py`

The model-fitting code chooses between them in:

- `CapstoneBO+/models/warped_gp.py`

### Why transform `y` at all?

Gaussian process models often fit better when the target values are less skewed and more regular.

If raw `y0` values are heavily skewed, a transform can make modeling easier.

### `OutputPowerTransform`

This class uses scikit-learn's `PowerTransformer`.

Its logic is:
- if all targets are positive, use Box-Cox,
- if all are negative, flip the sign, use Box-Cox, then remember to flip back later,
- otherwise use Yeo-Johnson,
- if the dataset is tiny, fall back to identity.

This is a nice beginner example of a stateful helper class:
- `fit(...)` learns parameters from the observed data,
- `transform(...)` applies the forward transform,
- `inverse_transform(...)` converts model-space values back into raw-space estimates.

### `OutputCopulaTransform`

This is the alternate transform mode.

It uses scikit-learn's `QuantileTransformer` to map observed targets to something closer to a standard normal distribution.

This can be useful when the target shape is awkward for a simple power transform.

### Important concept

After transformation, the GP is trained on transformed targets, not the raw `y0` values.

That means:
- `mu_t`, `sigma_t`, `ei`, `pi`, and `ucb` are mostly in transformed-target space,
- and later the code uses `inverse_transform(...)` to estimate raw-space means for reporting.

This is one of the most important details in the whole repository.

## 9. Stage 7: Optional Input Warping

Still in `models/warped_gp.py`, the code also optionally warps the inputs.

### What problem is this trying to solve?

Sometimes the response changes very unevenly across the input space. A warp can help the model represent that more naturally.

### `_make_input_warp(...)`

This helper tries to build a BoTorch `Warp` transform if the installed BoTorch version supports it.

If that is not available, it falls back to:

```python
class KumaraswamyWarpFallback(InputTransform):
    ...
```

### What the fallback class is doing

It defines a learnable transformation of each input dimension:

```python
w(x) = 1 - (1 - x^a)^b
```

You do not need to memorize the formula. The important idea is:
- it bends the input space,
- the parameters `a` and `b` are learned,
- and the GP sees the warped coordinates instead of the raw ones.

### Why there are so many compatibility checks

The BoTorch `Warp` API can differ between versions. The code uses `inspect.signature(...)` and several constructor attempts to stay compatible with different environments.

That is a practical engineering choice: the author wants the pipeline to degrade gracefully instead of breaking because one library version moved a keyword argument.

## 10. Stage 8: Choose a Kernel and Fit the GP

The kernel helper lives in:

- `CapstoneBO+/models/kernels.py`

The fitting function lives in:

- `CapstoneBO+/models/warped_gp.py`

### Kernel choice

`make_covar_module(...)` currently supports two modes:

- `baseline`
- `lin_matern32_add`

In plain English:
- `baseline` is a scaled Matérn kernel,
- `lin_matern32_add` is a scaled sum of a linear kernel and a Matérn 1.5 kernel.

The second option lets the model represent both linear trends and rough nonlinear structure.

### `_fit_warped_gp_once(...)`

This helper performs one actual model fit attempt. It:

1. Builds the output transform.
2. Applies it to the target.
3. Builds the input transform if enabled.
4. Builds the covariance module from the chosen kernel mode.
5. Creates a BoTorch `SingleTaskGP`.
6. Wraps it in an exact marginal log likelihood.
7. Optimizes the model parameters with `fit_gpytorch_mll(...)`.

### `FittedWarpedGP`

The fit result is stored in another dataclass:

```python
@dataclass
class FittedWarpedGP:
    model: SingleTaskGP
    y_transform: object
    y_raw: torch.Tensor
    y_t: torch.Tensor
    warp_kind: str
    ...
```

This gives the rest of the pipeline easy access to:
- the trained model,
- the transform object,
- the transformed targets,
- and diagnostics about how the fit was achieved.

### The fallback ladder

This is one of the most practical parts of the whole codebase.

The fitting function does not trust one single model setup to work every time. Instead, `fit_warped_gp(...)` tries a sequence of options:

1. primary settings
2. no input warp
3. no output power transform
4. baseline kernel
5. baseline kernel with no transforms
6. higher jitter
7. higher jitter plus baseline/no-transforms

This matters because GP fitting can be numerically fragile. Instead of crashing immediately, the pipeline tries simpler or safer model versions until one works.

This is why the code can say:

"The preferred model is warped GP with transforms, but the live run may fall back to a simpler configuration."

## 11. Stage 9: Build the Candidate Pool

Candidate generation lives in:

- `CapstoneBO+/candidates/pool.py`

The main function is:

```python
build_candidate_pool(...)
```

Its return type is:

```python
@dataclass
class CandidatePool:
    X_pool: torch.Tensor
    scores: AcquisitionScores
    priority_indices: np.ndarray
    candidate_origin: np.ndarray
    tr_state: TrustRegionState | None
```

### What the candidate pool is

This is the list of possible next points the pipeline is willing to consider before final selection.

The pipeline does not directly optimize the acquisition function with gradients. Instead, it:
- samples a large set of candidate points,
- scores them all,
- then chooses from that scored pool.

### Where do candidate points come from?

There are up to three sources:

1. `global`
   - a Sobol sample across the whole search box
2. `tr`
   - a Sobol sample inside the reconstructed trust region
3. `elite`
   - a Sobol sample inside a box built around the best observed region

### Why Sobol sampling?

Sobol points are a low-discrepancy sequence. You can think of them as a smarter way to spread points out than naive random sampling.

That makes them a good fit for "cover the box fairly well without needing too many points."

### The elite box

The elite pool is built from the best observed points so far:
- choose a top fraction of observed rows,
- take the coordinate-wise min and max,
- expand that box a bit with `elite_margin`,
- sample inside it.

This gives the pipeline another local-ish candidate source that is not identical to the trust-region box.

### `candidate_origin`

The code stores where each candidate came from:
- `"global"`
- `"tr"`
- `"elite"`

This becomes useful later for:
- reporting,
- trust-region-biased strategies,
- and understanding why a candidate was chosen.

## 12. Stage 10: Score Every Candidate

Acquisition scoring lives in:

- `CapstoneBO+/acquisition/scores.py`

The main result container is:

```python
@dataclass
class AcquisitionScores:
    mu_t: np.ndarray
    sigma_t: np.ndarray
    ei: np.ndarray
    log_ei: np.ndarray
    pi: np.ndarray
    ucb_by_beta: dict[float, np.ndarray]
    ucb_max: np.ndarray
    ucb_max_beta: np.ndarray
    maxvar: np.ndarray
    ts: np.ndarray
    ts_max: np.ndarray
```

If you were asking earlier what a class like this is doing, this is a perfect example:
- it does not itself perform the scoring,
- it just stores all the scoring outputs in one organized object.

### Step 10.1: Get the GP posterior

Inside `compute_acq_scores(...)`, the code first calls:

```python
post = model.posterior(X_pool)
```

That produces the GP's belief at every candidate point.

From that posterior, the code extracts:
- `mu_t`: predicted mean
- `sigma_t`: predicted standard deviation

Again, those are in transformed-target space.

### Step 10.2: Compute acquisition functions

The code then computes several scoring rules:

- `EI`: Expected Improvement
- `logEI`: logarithm of EI, used for better numerical stability
- `PI`: Probability of Improvement
- `UCB(beta)`: Upper Confidence Bound at many beta values
- `ts` / `ts_max`: Thompson-style samples

### Why use several acquisition functions instead of one?

Because each one emphasizes something slightly different:

- `mu_t` likes high predicted value
- `sigma_t` likes uncertainty
- `EI` balances improvement against uncertainty
- `PI` asks "how likely is improvement?"
- `UCB` explicitly trades mean and uncertainty
- `TS` injects randomness that can encourage useful exploration

Using several of them gives the pipeline more than one lens on the same candidate set.

### Step 10.3: Handle numerical issues carefully

The code prefers `LogExpectedImprovement` when available. Then it:
- exponentiates back to EI if needed,
- repairs invalid EI values,
- clips or sanitizes NaN and infinite values.

This is practical defensive programming.

In numerical ML code, "compute the fancy formula" is often only half the job. The other half is "make sure it does not explode on real data."

### Step 10.4: Record the best UCB per candidate

Because UCB is computed at a sweep of beta values, the code stores:
- `ucb_by_beta`: every beta's scores
- `ucb_max`: the best UCB value each candidate achieved across the sweep
- `ucb_max_beta`: which beta produced that best value

So each candidate gets both a best UCB score and the beta that made it look best.

## 13. Stage 11: Mark High-Priority Candidates

Still in `candidates/pool.py`, after the pool is scored, the code builds `priority_indices`.

It does this by taking the union of the top `k` indices from:
- `log_ei`
- `ei`
- `pi`
- `maxvar`
- `ts_max`
- and each `ucb(beta)`

### Why do this?

Right now, this is mostly metadata. The full pool is still kept for final selection.

But it is useful because it identifies points that look good under at least one acquisition rule.

This is a sign of extensible design:
- the pipeline does not fully depend on `priority_indices` yet,
- but it stores them because they may become useful for later filtering, reporting, or future heuristics.

## 14. Stage 12: Remove Portal Duplicates After Rounding

This logic lives in:

- `CapstoneBO+/rounding.py`

This is one of the most domain-specific parts of the pipeline.

### Why this step exists

The portal does not accept full floating-point precision. The submission gets rounded to a fixed number of decimals.

That means two candidates that look different in raw floating-point form might become the same portal string after rounding.

Even worse, a new candidate might round to something you already observed before.

So the pipeline must deduplicate after rounding, not before.

### `quantize_array_floor(...)`

This helper:
- clips values into the legal range,
- multiplies by `10**decimals`,
- applies `floor`,
- divides back down.

This is floor-based quantization, not standard round-to-nearest.

### `format_portal_string(...)`

This converts one candidate row into the final hyphen-separated text format expected by the portal.

### `dedup_after_rounding(...)`

This function:

1. Quantizes candidate points.
2. Quantizes observed points.
3. Drops candidates that collide with already-observed quantized points.
4. Resolves internal candidate collisions by keeping only the best-scoring one.

The scoring signal used for those collisions in `weekly_pack.py` is:

```python
shadow_score = mu_t + 0.5 * sigma_t
```

That is important:
- the shadow score is not necessarily the final submission rule,
- but it is used as the tie-break when duplicate candidates collapse to the same portal key.

### Why the fallback candidate exists

In rare cases, deduplication can remove every candidate.

If that happens, `_make_random_nonduplicate(...)` creates a random legal point that does not collide after rounding.

That is another practical safety net: the pipeline would rather return a safe fallback than crash or return an illegal duplicate.

## 15. Stage 13: Compute Extra Selection Helpers

Back in `weekly_pack.py`, after deduplication, the code computes some extra metrics that are not standard BO formulas but are very useful operationally.

### Boundary penalty

The helper functions are:
- `_boundary_prox(...)`
- `_boundary_penalty(...)`

These measure how close a candidate is to the boundary of the input box.

For higher-dimensional problems, the code can reduce the tie-break mean:

```python
mu_t_adj = mu_t - boundary_penalty
```

That means a point with strong predicted mean can be gently downgraded if it hugs the edge too aggressively.

This is not a hard constraint. It is a soft preference.

### Novelty

The helper `_nearest_distance(...)` computes the distance from each candidate to the closest observed point.

Then the code compares that distance to a typical observed nearest-neighbor distance and builds:

- `novelty_ratio`

High novelty means the candidate is geometrically farther from known data.

### Exploration score

The code defines:

```python
exploration_score = sigma_t * novelty_ratio
```

This is an explicit "blind spot" score.

A point is interesting for exploration when it is:
- uncertain, and
- far from already-sampled points.

That is a more useful exploration signal than uncertainty alone, because high uncertainty near already-known points is not always that interesting.

## 16. Stage 14: Select the Final Candidate

Selection lives in:

- `CapstoneBO+/candidates/selectors.py`
- `CapstoneBO+/acquisition/mace_pareto.py`

This is where many of the strategic choices come together.

### Step 14.1: Build a Pareto problem

`select_submission_from_pareto(...)` creates a minimization objective matrix:

```python
(-ei, -pi, -ucb)
```

Why the negative signs?

Because the Pareto helper is written in minimization form, but EI, PI, and UCB are scores you normally want to maximize. Negating them turns "bigger is better" into "smaller is better."

### Step 14.2: Find the Pareto front

The helpers in `acquisition/mace_pareto.py` compute:
- which points are non-dominated,
- and which Pareto rank each point belongs to.

Beginner translation:

A point is on the Pareto front if no other point is at least as good in every objective and strictly better in at least one.

This is a way to keep candidates that are strong in different ways, instead of forcing everything through one combined score too early.

### Step 14.3: Apply the sigma gate

Among Pareto-front candidates, the code keeps only those above a sigma threshold determined by `sigma_quantile_schedule(n_points)`.

This is a built-in exploration guard.

Even after Pareto filtering, the pipeline does not want to choose a very low-uncertainty point too quickly if the uncertainty structure suggests broader exploration is still useful.

### Step 14.4: Tie-break by mean

After the sigma gate, the code selects by:
- highest `mu_t_tiebreak`
- then highest `EI`
- then highest `PI`
- then smallest index for deterministic behavior

Usually `mu_t_tiebreak` is just `mu_t`, but in higher dimensions it can be the boundary-penalized version `mu_t_adj`.

### Step 14.5: Build the portfolio choices

The next helper is:

```python
select_portfolio(...)
```

This function creates a small family of named strategies:

- `primary_pareto_sigma`
- `global_explore`
- `shadow_oldstyle`
- `trust_region_ts`

#### What each one means

- `primary_pareto_sigma`
  - the main Pareto + sigma-gate selection rule
- `global_explore`
  - choose the non-trust-region candidate with the highest `exploration_score` when possible
- `shadow_oldstyle`
  - choose by `mu_t + 0.5 * sigma_t`
- `trust_region_ts`
  - prefer a high Thompson-style score inside the trust-region subset if possible

### Step 14.6: Let stagnation choose the behavior

The final selected strategy name is:

- `cfg.portfolio_primary` when not stagnant
- `cfg.portfolio_when_stagnant` when stagnant

So the pipeline can say, in effect:

"When progress looks healthy, use the main Pareto rule. When progress looks stuck, switch to a more trust-region or exploration-aware option."

The default stagnant mode is now a hybrid:

- if the best trust-region candidate still has competitive optimistic upside, stay local with `trust_region_ts`
- otherwise jump to `global_explore`

This is a very important behavioral feature of `CapstoneBO+`.

## 17. Stage 15: Build the Report Rows

Once the final candidate is chosen, `weekly_pack.py` builds several kinds of output rows.

### Candidate rows

For every eligible candidate after deduplication, the pipeline stores a row containing fields such as:
- portal string,
- `mu_t`,
- `sigma_t`,
- `ei`,
- `pi`,
- `ucb`,
- Pareto rank,
- novelty metrics,
- boundary metrics,
- origin (`global`, `tr`, `elite`, or fallback),
- raw and quantized coordinates.

This becomes `candidate_summary.csv`.

### Recommendation rows

The pipeline also creates a shorter table with the named strategy anchors:
- chosen submission,
- best mean,
- best sigma,
- best logEI,
- best PI,
- best UCB,
- best exploration,
- shadow old-style,
- and so on.

This becomes `recommendations.csv`.

### Diagnostics payload

The code also combines model diagnostics and run diagnostics:
- fit attempt label,
- failed attempts,
- selected strategy,
- trust-region length,
- duplicate counts,
- warning counts,
- and more.

This becomes one JSON file per function.

### Why so much reporting?

Because this is not just a black-box suggestion engine. It is also an inspection tool.

A lot of the code is designed so you can answer questions like:
- Why was this point chosen?
- Was the model stable?
- Did several seeds agree?
- Was the chosen point from the global pool or a local pool?
- Did deduplication remove many candidates?

## 18. Stage 16: Aggregate Across Seeds

If multiple seeds were run, `run_weekly(...)` then performs consensus selection.

### `_consensus_pick(...)`

This helper groups successful seed results by `portal_string`.

For each portal string, it computes:
- frequency,
- mean submission score,
- mean `mu_t`.

Then it ranks them:
- frequency first,
- then tie-break by mean score or mean `mu_t` depending on config,
- then lexical order as a final deterministic tie-break.

### Why compare portal strings and not raw floats?

Because the portal string is the actual submitted object. Two different raw floating-point candidates that round to the same portal string are operationally the same submission.

That is why consensus is applied to the rounded/formatted output, not to raw coordinates.

### Fallback behavior

If too few seeds succeed, the pipeline can fall back to:
- seed 0 if available,
- otherwise the first successful seed.

This keeps the system useful even when some seed runs fail.

### Representative rows come from one selected seed

An important detail:

The final `candidate_summary.csv` and `recommendations.csv` do not merge all seed pools together.

Instead, they use the candidate and recommendation rows from the seed run that represents the selected final choice.

That is simpler to inspect and avoids mixing incompatible candidate sets from different randomized runs.

## 19. Stage 17: Write Files to Disk

Reporting code lives in:

- `CapstoneBO+/reporting/export_csv.py`
- `CapstoneBO+/reporting/dashboard.py`

### `write_report_csvs(...)`

This function writes:
- `candidate_summary.csv`
- `recommendations.csv`
- `portal_strings.csv`
- `seed_suggestions.csv`
- `stability_summary.csv`
- `run_health.json`
- per-function `model_diagnostics.json`

It also creates:
- `candidate_shortlist.csv`
- `candidate_pivot.csv`

These are convenience files for easier comparison and diffing.

### `make_weekly_dashboard(...)`

This builds a lightweight HTML page with:
- a recommendations table,
- a candidate summary table.

This is not a complex web app. It is a simple static report writer. That is why the code can stay short and readable.

## 20. What the Final Output Files Mean

Here is the practical meaning of the main report files.

### `portal_strings.csv`

The final answer table.

If you only want the chosen submission string for each function, this is the most direct file.

### `recommendations.csv`

A compact table of important named picks.

Use this when you want to compare:
- what the main selector chose,
- what a more exploratory rule would have chosen,
- what the best-UCB or best-EI point looked like,
- and whether the final submission is aligned with or different from those alternatives.

### `candidate_summary.csv`

The full inspected pool for the representative seed.

Use this when you want to understand:
- all eligible candidates,
- Pareto ranks,
- uncertainty,
- novelty,
- boundary effects,
- and candidate origins.

### `stability_summary.csv`

The high-level multi-seed stability report.

Use this when you want to know:
- how many seeds succeeded,
- how many unique suggestions appeared,
- how strong the consensus was,
- whether fallback was needed.

### `run_health.json`

The run-wide health summary.

Use this when you want seed-level failure counts, warning counts, and overall failure rate.

### `function_k/model_diagnostics.json`

The detailed model report for one function.

Use this when you want fitting details such as:
- which fallback attempt actually worked,
- noise estimate,
- lengthscale summary,
- warp parameters,
- trust-region summary,
- selection metadata.

## 21. A Short "One Function" Story

Sometimes it helps to compress the whole flow into one narrative.

For one function, one seed, the pipeline does this:

1. Load observed `x` and `y0` values from CSV.
2. Transform `y0` into a modeling-friendly version.
3. Build a GP that may warp the inputs and may use one of two kernels.
4. If fitting fails, try simpler fallback versions.
5. Reconstruct a trust-region state and detect whether recent progress looks stagnant.
6. Sample many candidate points from global, trust-region, and elite boxes.
7. Ask the GP what it thinks about every candidate.
8. Turn those beliefs into acquisition scores like EI, PI, UCB, and TS.
9. Remove candidates that would become illegal duplicates after portal rounding.
10. Compute novelty, exploration, and boundary penalties.
11. Pick a final point using Pareto filtering, uncertainty gating, and a strategy portfolio.
12. Format that point into a portal-safe string.
13. Save candidate tables, recommendation tables, diagnostics, and health summaries.

That is the whole weekly pipeline in plain English.

## 22. Why the Code Uses So Many Small Helper Functions

If you are newer to reading code, you may wonder why the pipeline is split into many small functions instead of one giant script.

There are three main reasons:

### Reason 1: easier testing and debugging

It is much easier to debug:
- `dedup_after_rounding(...)`
than a giant 700-line function where deduplication is buried in the middle.

### Reason 2: clearer data flow

Small functions make it easier to see:
- what goes in,
- what comes out,
- and where the next stage gets its inputs.

### Reason 3: safer future edits

If you want to improve one stage, you can often do it without rewriting the entire pipeline.

For example:
- change kernels in `models/kernels.py`,
- change acquisitions in `acquisition/scores.py`,
- change portfolio logic in `candidates/selectors.py`,
- change report formatting in `reporting/export_csv.py`.

That is a sign of decent modular design.

## Appendix A: The High-Dimensional Summary Plot Script

This appendix is intentionally short. The main weekly optimization pipeline is the real focus of this document.

The file:

- `CapstoneBO+/plot_highdim_function_summaries.py`

creates the 2x3 summary PNGs for higher-dimensional functions.

At a high level, it:
- loads one dataset,
- computes PCA and t-SNE views of the observed inputs,
- fits the same GP pipeline,
- creates a 2D GP mean slice over two selected features,
- plots 1D partial dependence curves,
- and adds a target-distribution/statistics panel.

This script is mainly for human interpretation. It does not choose the weekly submission.

In other words:
- `weekly_pack.py` is about decision-making,
- `plot_highdim_function_summaries.py` is about visualization.

## Appendix B: The 2D Surface Plot Script

The file:

- `CapstoneBO+/plot_function_surfaces.py`

is a simpler plotting tool for low-dimensional functions, especially `function_1` and `function_2`.

It:
- reads `x0`, `x1`, and `y0`,
- fits a smooth interpolation surface with `RBFInterpolator`,
- renders a 3-panel image:
  - 3D scatter,
  - contour plot,
  - fitted 3D surface.

This script is also for understanding the data, not for selecting the weekly point.

## Appendix C: Python Patterns You Will See in This Repository

### Dataclasses

You have already seen several:
- `RebuildConfig`
- `FunctionDataset`
- `FittedWarpedGP`
- `CandidatePool`
- `AcquisitionScores`
- `SelectionResult`
- `FunctionSeedResult`

These are mostly structured containers for related values.

### Type hints

Examples:

```python
def load_function_dataset(csv_path: Path, dtype: torch.dtype = torch.double) -> FunctionDataset:
```

The `-> FunctionDataset` part is a type hint. It tells the reader what should come back from the function.

Type hints are mainly for readability, editors, and static analysis. They do not usually force runtime behavior by themselves.

### Leading underscores

Functions like `_run_one_function(...)` or `_nearest_distance(...)` start with `_`.

That is a Python naming convention meaning:

"This is an internal helper, not part of the public API."

Python does not truly hide it, but the name signals intent.

### `if __name__ == "__main__":`

This appears in the scripts so that the file can be:
- imported as a module, or
- run as a script.

When you run the file directly, the `main()` function is called.

### Moving between pandas, NumPy, and PyTorch

This repository frequently moves between three data containers:

- `pd.DataFrame`
  - best for labeled tabular data and CSV I/O
- `np.ndarray`
  - best for lightweight numeric manipulation
- `torch.Tensor`
  - best for model fitting with BoTorch/GPyTorch

You will often see lines like:

```python
X_np = X_keep.detach().cpu().numpy()
```

That means:
- detach from the PyTorch computation graph,
- move to CPU memory,
- convert to a NumPy array.

This pattern is very common in scientific Python code.

## Final Mental Model

If you want a compact mental model for the whole repository, use this:

- `config.py` says how the run should behave.
- `data_io.py` turns CSV files into clean tensors.
- `models/warped_gp.py` turns historical observations into a GP model.
- `candidates/pool.py` generates points worth considering.
- `acquisition/scores.py` scores those points in several ways.
- `rounding.py` makes sure the portal output is legal and non-duplicate.
- `candidates/selectors.py` chooses one point from the scored pool.
- `weekly_pack.py` orchestrates all of it and writes the results.

That is CapstoneBO+ in one paragraph.
