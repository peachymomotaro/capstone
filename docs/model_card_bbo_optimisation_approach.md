# Model Card for the CapstoneBO+ Optimisation Approach

## Overview

**Name.** CapstoneBO+ portfolio-based weekly black-box optimisation pipeline.

**Type.** Sequential surrogate-based optimiser for continuous box-constrained black-box functions.

**Version label used here.** This model card describes the live `CapstoneBO+` code in [`config.py`](../config.py), [`weekly_pack.py`](../weekly_pack.py), and related modules, together with the stored report variants through [`reports/2026-04-02-fixed-signflip/`](../reports/2026-04-02-fixed-signflip). No git revision is available in this workspace copy, so the dated report folders are the most concrete revision markers.

**Short summary.** CapstoneBO+ fits one Gaussian process per function, transforms outputs when helpful, optionally warps inputs, generates a mixed candidate pool from global, trust-region, and elite regions, scores those candidates with multiple acquisition signals, removes portal-rounding duplicates, and chooses one weekly submission per function through a small portfolio of decision rules. In the default workflow it repeats that process across five seeds and reports a consensus portal string.

## Intended Use

CapstoneBO+ is suitable for the setting actually represented in this repository:

- one expensive black-box function at a time;
- continuous inputs already normalised to approximately `[0, 1)`;
- single-point weekly submissions rather than batches;
- no categorical variables, no explicit constraints, and no external cost model;
- historical observations available as cumulative CSV files.

The expected input is a per-function history with columns `x0`-`xk` and `y0`. The expected output is a portal-formatted candidate string plus supporting report tables and diagnostics.

Use cases to avoid include:

- treating the pipeline as a generic optimiser for arbitrary domains without rescaling;
- applying it to constrained or categorical search spaces without modification;
- interpreting it as a classifier, regressor, or fairness-sensitive human-decision model;
- claiming guaranteed global optimality from the returned suggestions.

The valid application scope is therefore fairly narrow: small-to-moderate-dimensional continuous capstone functions, chosen one query at a time, with heavy emphasis on robustness and auditability.

## Approach Details

### Core pipeline

The live code path in [`weekly_pack.py`](../weekly_pack.py) runs the following per function:

1. Load the cumulative history from `function_k.csv`.
2. Apply per-function overrides. The only explicit hard-coded override is for `function_1`, where the output power transform is disabled and both portfolio modes are forced to `global_explore`.
3. Fit a GP with [`models/warped_gp.py`](../models/warped_gp.py):
   - default kernel `lin_matern32_add` (`Linear + Matérn 1.5`, ARD, scaled);
   - optional input warp via BoTorch `Warp`, with a Kumaraswamy fallback if needed;
   - default output transform mode `power`, implemented through Box-Cox, sign-flipped Box-Cox, or Yeo-Johnson.
4. Reconstruct a trust region from the historical sequence with [`candidates/trust_region.py`](../candidates/trust_region.py).
5. Build a mixed candidate pool with [`candidates/pool.py`](../candidates/pool.py):
   - `3072` global Sobol points by default;
   - `1024` trust-region Sobol points if trust-region mode is enabled;
   - `512` elite-region Sobol points from a box around the top `20%` observed points.
6. Score every candidate in [`acquisition/scores.py`](../acquisition/scores.py) using:
   - posterior mean and standard deviation in transformed space;
   - EI, logEI, PI;
   - a sweep of UCB values with betas from `0.50` to `3.00`;
   - Thompson-style scores `ts` / `ts_max`;
   - a later exploration score computed as uncertainty times novelty.
7. Remove internal duplicates and collisions with observed points **after** six-decimal floor rounding via [`rounding.py`](../rounding.py).
8. Choose the final candidate with [`candidates/selectors.py`](../candidates/selectors.py), using named strategies:
   - `primary_pareto_sigma`;
   - `trust_region_ts`;
   - `global_explore`;
   - `shadow_oldstyle`.
9. Repeat across seeds `0`-`4` in robust-consensus mode and choose one final portal string per function.

### Surrogate modelling choices

The modelling layer is best understood as “preferred configuration plus robust fallback chain,” not as one fixed model. The preferred path is:

- transformed target;
- input warp;
- additive linear plus Matérn kernel;
- exact GP fit in double precision.

If that fails, the code falls back through no-warp, no-transform, baseline-kernel, and high-jitter variants. That fallback chain matters because it makes the pipeline more operationally robust, but it also means the phrase “CapstoneBO+ uses warped GP with power transform” is only conditionally true. In the stored `2026-04-02*` diagnostics, all eight functions happened to fit on the primary attempt.

The codebase also contains some options that are **present but not primary**:

- `output_transform_mode="copula"` is implemented, but the examined stored diagnostics only show `identity`, `yeo-johnson`, or `box-cox`;
- config and CLI flags mention `enable_model_ensemble`, but the current code inspection did not find an active ensemble-scoring path in the runtime pipeline.

Those options should therefore be described as available hooks, not as confirmed components of the capstone deliverables.

### Selection logic and exploration/exploitation trade-off

The primary selector still reflects the older CapstoneBO philosophy: compute a Pareto front over `(-EI, -PI, -UCB)`, apply a sigma quantile gate that becomes less strict with more observations, then tie-break by posterior mean. That is the `primary_pareto_sigma` branch.

CapstoneBO+ adds two important alternatives:

- `trust_region_ts`: favour a local trust-region candidate with a Thompson-style score;
- `global_explore`: favour geometrically novel, high-uncertainty global candidates.

The exploration score is not a standard acquisition function but a repository-specific diagnostic:

`exploration_score = sigma_t * novelty_ratio`

where `novelty_ratio` is the distance to the nearest observed point divided by the median nearest-neighbour distance among observed points.

For higher-dimensional functions, the final tie-break mean is additionally penalised near the search-space boundary. This does not change the Pareto front itself; it only changes the final ranking among eligible points.

### How the approach evolved in this repository

The repository tells an evolution story in three layers:

1. [`Handover.md`](../Handover.md) still documents the older global-Sobol CapstoneBO pipeline.
2. [`Recommendations.md`](../Recommendations.md) proposes trust-region and portfolio upgrades inspired by external BBO competition ideas.
3. The live `CapstoneBO+` code and report files show that those trust-region and portfolio ideas were in fact implemented.

That means the most defensible interpretation is:

- early work started from a global Pareto-plus-sigma BO baseline;
- CapstoneBO+ then added trust-region reconstruction, elite-region sampling, Thompson-style scoring, stagnation detection, and explicit portfolio switching;
- the dated reports on `2026-03-19-plus`, `2026-03-20`, and `2026-04-02*` record the approach being used on accumulating data.

The repository also contains a subtle reproducibility issue: the dataclass default in [`config.py`](../config.py) sets `portfolio_when_stagnant="trust_region_ts"`, but the CLI in [`weekly_pack.py`](../weekly_pack.py) defaults that argument to `stagnation_hybrid`. Because the stored `2026-04-02` reports choose `global_explore` for one stagnating function and `trust_region_ts` for another, the report evidence suggests the CLI-style hybrid setting was used for those runs.

### About the requested “ten rounds”

The repository strongly suggests a weekly sequential capstone process, but it does **not** expose a clean round-by-round manifest for all ten rounds. What can be reconstructed reliably is:

- cumulative raw-data growth across dated report folders;
- a weekly append workflow from the notebook and code;
- one additional raw observation per function between some dated snapshots.

It would therefore be overstated to pretend that the exact logic of each of ten rounds can be read off directly from the files. Based on the available reports, the approach by late March and early April 2026 was already the mixed-pool, trust-region, portfolio-based CapstoneBO+ described above.

## Performance

There is no single authoritative offline score stored in this repository. The strongest evidence comes from two different sources:

1. the **raw data files**, which show the best observed `y0` reached so far for each function;
2. the **dated reports**, which show which strategy and candidate origin the optimiser selected at each snapshot.

These are not the same thing. Reported `mu_t`, `sigma_t`, and `ucb` values are in transformed-target space, so they are not directly comparable with raw `y0` across functions. For negative-output problems such as `function_3` and `function_6`, transformed-space means can even be positive after sign-flipped Box-Cox processing. Because of that, the most interpretable cross-function performance summary is the raw best observed objective in the cumulative dataset.

Using the current raw data together with the latest matching report variant (`2026-04-02-fixed-signflip`, identical to `2026-04-02-audited` on chosen submissions):

| Function | d | n | Best observed raw `y0` | Best-so-far improvement over stored run | Latest chosen strategy / origin | Stagnation flag in latest report |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `function_1` | 2 | 19 | `6.83e-12` | `+6.83e-12` | `global_explore` / `global` | No |
| `function_2` | 2 | 19 | `0.651454` | `+0.112458` | `primary_pareto_sigma` / `tr` | No |
| `function_3` | 3 | 24 | `-0.004284` | `+0.107838` | `trust_region_ts` / `tr` | Yes |
| `function_4` | 4 | 39 | `0.608628` | `+22.716916` | `trust_region_ts` / `tr` | Yes |
| `function_5` | 4 | 29 | `6049.023450` | `+5984.580010` | `primary_pareto_sigma` / `tr` | No |
| `function_6` | 5 | 29 | `-0.162953` | `+0.551312` | `trust_region_ts` / `tr` | Yes |
| `function_7` | 6 | 39 | `2.244001` | `+1.639568` | `trust_region_ts` / `tr` | Yes |
| `function_8` | 8 | 49 | `9.920822` | `+2.522101` | `trust_region_ts` / `tr` | Yes |

The most defensible summary is therefore mixed:

- the cumulative capstone trajectory shows substantial best-so-far gains on several functions, especially `function_4`, `function_5`, and `function_8`;
- the stored late-stage reports show the optimiser leaning heavily toward trust-region picks: in `2026-04-02-fixed-signflip`, seven of eight chosen submissions came from the trust-region pool;
- five of eight functions were flagged as stagnating in that latest report, which is consistent with the selector shifting toward `trust_region_ts` under the hybrid stagnant-policy logic.

There is also a robustness caveat that should not be hidden: in the latest stored stability summary, **all eight functions achieved 5/5 successful seed runs, but every function had a consensus frequency ratio of `0.2`**. In practice, that means each of the five seeds selected a different portal string, so the final “consensus” was resolved by frequency ties plus tie-break rules rather than true agreement. Operational reliability was good; seed-to-seed reproducibility of the exact recommendation was not.

The latest run-health files also report `70` negative-variance warnings globally and zero optimisation failures. That suggests the pipeline remained numerically usable, but not numerically pristine.

## Assumptions and Limitations

Core assumptions include:

- maximisation rather than minimisation;
- continuous box-constrained inputs already scaled near `[0, 1)`;
- chronological row order as a meaningful history for trust-region reconstruction;
- surrogate smoothness that is at least approximately capturable by the chosen GP kernels and transforms.

Important limitations and brittleness include:

- **small data per function**: even the largest function has only 49 observations;
- **policy dependence**: later data were chosen by earlier optimisers, so performance summaries are not unbiased;
- **rounding effects**: six-decimal floor quantisation can collapse distinct floating-point candidates into the same portal point;
- **boundary effects**: the boundary penalty is heuristic rather than principled;
- **transform sensitivity**: sign-flipped Box-Cox helps all-negative objectives, but it also makes transformed-space diagnostics harder to interpret;
- **trust-region reconstruction**: the trust region is inferred from past observations rather than stored as a native optimiser state;
- **seed instability**: the 0.2 consensus ratio in the latest reports indicates materially different suggestions across seeds;
- **configuration ambiguity**: stored reports appear to rely on a CLI default (`stagnation_hybrid`) that differs from the dataclass default (`trust_region_ts`).

There is also an historical limitation: the cumulative dataset spans multiple stages of the capstone, so the raw best-so-far improvements cannot be attributed cleanly to CapstoneBO+ alone without a stricter provenance log.

## Ethical Considerations

This is not a human-subject model card, so the most relevant ethical issues are transparency, reproducibility, and responsible claims.

The repository is relatively strong on operational transparency: it stores candidate-level tables, named recommendation strategies, seed-level summaries, run-health JSON, and per-function model diagnostics. That makes auditing easier than in many student optimisation projects.

The main ethical risk is overclaiming. Because the problem is sequential, small-sample, and policy-dependent, it would be irresponsible to present these results as a clean, general proof that this approach is broadly superior to other BO methods. Responsible reuse means stating clearly which parts are directly evidenced by code and outputs, and which parts are inferences from dated artifacts.

## Why This Level of Detail Is Sufficient

For this repository, more detail would not automatically produce more clarity. The main modelling and decision choices are already identifiable from the code:

- which transforms are used;
- how candidates are generated;
- how the selector switches under stagnation;
- which diagnostics are logged.

What the repository does **not** provide is an exact command log, git revision, or full per-round provenance. More prose cannot fill that gap honestly. The current level of detail is therefore sufficient for peer-review-style understanding of the implemented approach, while still acknowledging the remaining unknowns.

The main documentation gaps are:

- no git hash or formal version tag in this workspace copy;
- no exact ten-round manifest;
- no explicit statement of which `2026-04-02*` report variant is authoritative;
- no formal authorship or licence file.

## Reproducibility Notes

To understand or rerun the approach, inspect these files first:

- [`config.py`](../config.py) for defaults and per-function overrides;
- [`weekly_pack.py`](../weekly_pack.py) for orchestration, portfolio wiring, and consensus logic;
- [`candidates/pool.py`](../candidates/pool.py) for global/trust-region/elite candidate generation;
- [`candidates/trust_region.py`](../candidates/trust_region.py) for trust-region reconstruction;
- [`candidates/selectors.py`](../candidates/selectors.py) for portfolio choice and stagnant-mode logic;
- [`acquisition/scores.py`](../acquisition/scores.py) for EI/PI/UCB/TS scoring;
- [`models/warped_gp.py`](../models/warped_gp.py) and [`transforms/output_power.py`](../transforms/output_power.py) for the surrogate and transforms;
- [`data/function_*.csv`](../data) for the cumulative optimisation history;
- [`reports/2026-04-02-fixed-signflip/`](../reports/2026-04-02-fixed-signflip) and [`reports/2026-04-02-audited/`](../reports/2026-04-02-audited) for the latest report variants aligned with the current raw data;
- [`tests/test_stagnation_and_portfolio.py`](../tests/test_stagnation_and_portfolio.py) for explicit tests of the low-signal override and stagnation portfolio logic.
