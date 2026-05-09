# Model Card for the CapstoneBO+ Optimisation Approach

## Overview

**Name.** CapstoneBO+ portfolio-based weekly black-box optimisation pipeline.

**Type.** Sequential surrogate-based optimiser for continuous box-constrained black-box functions.

**Short summary.** CapstoneBO+ fits one Gaussian process per function, transforms outputs when helpful, optionally warps inputs, generates a mixed candidate pool from global, trust-region, and elite regions, scores those candidates with multiple acquisition signals, removes portal-rounding duplicates, and chooses one weekly submission per function through a small portfolio of decision rules.

## Intended Use

CapstoneBO+ is used to optimise one expensive black-box function at a time:

It should take:
- continuous inputs already normalised to approximately `[0, 1)`;
- single-point weekly submissions rather than batches;
- no categorical variables, no explicit constraints, and no external cost model;
- historical observations available as cumulative CSV files.

The expected input is a per-function history with columns `x0`-`xk` and `y0`. The expected output is a portal-formatted candidate string plus supporting report tables and diagnostics.

## Approach Details

### Core pipeline

The live code path in [`weekly_pack.py`](../weekly_pack.py) runs the following per function:

1. Load the cumulative history from `function_k.csv`.
2. Apply per-function overrides. For now, the only explicit hard-coded override is for `function_1`, where the output power transform is disabled and both portfolio modes are forced to `global_explore`.
3. Fit a GP with [`models/warped_gp.py`](../models/warped_gp.py):
   - default kernel `lin_matern32_add` (`Linear + Matérn 1.5`, ARD, scaled);
   - optional input warp via BoTorch `Warp`, with a Kumaraswamy fallback if needed;
   - default output transform mode `power`, implemented through Box-Cox, sign-flipped Box-Cox, or Yeo-Johnson.
4. Reconstruct a trust region from the historical sequence with [`candidates/trust_region.py`](../candidates/trust_region.py).
5. Build a mixed candidate pool with [`candidates/pool.py`](../candidates/pool.py):
   - `3072` global Sobol points by default;
   - `1024` trust-region Sobol points;
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
9. Repeat across seeds `0`-`4` in robust-consensus mode and choose one final portal string per function.

### Surrogate modelling choices

The preferred path is:

- transformed target;
- input warp;
- additive linear plus Matérn kernel;
- exact GP fit in double precision.

### Selection logic and exploration/exploitation trade-off

The primary selector still reflects my older CapstoneBO philosophy. This involved computing a Pareto frontier over `(-EI, -PI, -UCB)`, applying a sigma quantile gate that becomes less strict with more observations, then tie-breaking by posterior mean. That is the `primary_pareto_sigma` branch.

CapstoneBO+ includes two important alternatives:

- `trust_region_ts`: favour a local trust-region candidate with a Thompson-style score;
- `global_explore`: favour geometrically novel, high-uncertainty global candidates.

The exploration score is calculated as:

`exploration_score = sigma_t * novelty_ratio`

where `novelty_ratio` is the distance to the nearest observed point divided by the median nearest-neighbour distance among observed points.

## Performance

Using the current raw CSV histories in [`data/`](../data), updated after the most recent appended observations:

| Function | d | n | Best observed raw `y0` | Best index | Latest raw `y0` | Submissions after best | 2026-04-30 chosen origin | 2026-04-30 stagnation flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `function_1` | 2 | 23 | `0.184524` | 22 | `0.184524` | 0 | `global` | No |
| `function_2` | 2 | 23 | `0.651454` | 17 | `0.617305` | 5 | `tr` | No |
| `function_3` | 3 | 28 | `-0.003253` | 27 | `-0.003253` | 0 | `tr` | Yes |
| `function_4` | 4 | 43 | `0.701305` | 41 | `0.357287` | 1 | `tr` | Yes |
| `function_5` | 4 | 33 | `8662.405001` | 31 | `8656.731627` | 1 | `global` | No |
| `function_6` | 5 | 33 | `-0.154283` | 32 | `-0.154283` | 0 | `tr` | Yes |
| `function_7` | 6 | 43 | `3.223213` | 39 | `3.203689` | 3 | `tr` | No |
| `function_8` | 8 | 53 | `9.958745` | 49 | `9.940839` | 3 | `tr` | No |

The latest raw observations improved the current best-so-far value for `function_1`, `function_3`, and `function_6`. The latest observation for `function_5` is very close to the current best but does not exceed it. The functions with active post-best submissions are now `function_2`, `function_4`, `function_5`, `function_7`, and `function_8`.

In the latest stored report folder, [`reports/2026-04-30/`](../reports/2026-04-30), all eight functions completed five successful seed runs under `robust_consensus`. The selected candidate origins were six trust-region submissions and two global submissions. Three of eight chosen submissions were marked as stagnating in that report: `function_3`, `function_4`, and `function_6`.

## Assumptions and Limitations

Core assumptions include:

- maximisation rather than minimisation
- inputs should be box-constrained within the unit box `[0, 1)`
- surrogate smoothness that is at least approximately capturable by the chosen GP kernels and transforms.

Important limitations include:

- **small data per function**: even the largest function has only 49 observations;
- **policy dependence**: later data were chosen by earlier optimisers, so performance summaries are not unbiased;
- **rounding effects**: six-decimal floor quantisation can collapse distinct floating-point candidates into the same portal point;
- **transform sensitivity**: sign-flipped Box-Cox helps all-negative objectives, but it also makes transformed-space diagnostics harder to interpret;
- **trust-region reconstruction**: the trust region is inferred from past observations rather than stored as a native optimiser state

## Ethical Considerations

This model card applies to fictional data, so any ethical issues are mainly around transparency.

## Why This Level of Detail Is Sufficient

The main modelling and decision choices are already identifiable from the code:

- which transforms are used
- how candidates are generated
- how the selector switches under stagnation
- which diagnostics are logged

The main documentation gaps are:

- no exact ten-round manifest.

## Reproducibility Notes

To understand this approach, inspect these files first:

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
