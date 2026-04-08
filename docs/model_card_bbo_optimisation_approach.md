# Model Card for the CapstoneBO+ Optimisation Approach

## Overview

**Name.** CapstoneBO+ portfolio-based weekly black-box optimisation pipeline.

**Type.** Sequential surrogate-based optimiser for continuous box-constrained black-box functions.

**Version label used here.** This model card describes the live `CapstoneBO+` code in [`config.py`](../config.py), [`weekly_pack.py`](../weekly_pack.py), and related modules, together with the stored report variants through [`reports/2026-04-02-fixed-signflip/`](../reports/2026-04-02-fixed-signflip). No git revision is available in this workspace copy, so the dated report folders are the most concrete revision markers.

**Short summary.** CapstoneBO+ fits one Gaussian process per function, transforms outputs when helpful, optionally warps inputs, generates a mixed candidate pool from global, trust-region, and elite regions, scores those candidates with multiple acquisition signals, removes portal-rounding duplicates, and chooses one weekly submission per function through a small portfolio of decision rules. In the default workflow it repeats that process across five seeds and reports a consensus portal string.

## Intended Use

CapstoneBO+ is used to optimise one expensive black-box function at a time:

It should take:
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
2. Apply per-function overrides. For now, the only explicit hard-coded override is for `function_1`, where the output power transform is disabled and both portfolio modes are forced to `global_explore`.
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

### Selection logic and exploration/exploitation trade-off

The primary selector still reflects my older CapstoneBO philosophy: compute a Pareto front over `(-EI, -PI, -UCB)`, apply a sigma quantile gate that becomes less strict with more observations, then tie-break by posterior mean. That is the `primary_pareto_sigma` branch.

CapstoneBO+ includes two important alternatives:

- `trust_region_ts`: favour a local trust-region candidate with a Thompson-style score;
- `global_explore`: favour geometrically novel, high-uncertainty global candidates.

The exploration score is calculated as:

`exploration_score = sigma_t * novelty_ratio`

where `novelty_ratio` is the distance to the nearest observed point divided by the median nearest-neighbour distance among observed points.

## Performance

Using the current raw data:

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

In summary:

- the cumulative capstone trajectory shows substantial best-so-far gains on several functions, especially `function_4`, `function_5`, and `function_8`;
- the stored late-stage reports show the optimiser leaning heavily toward trust-region picks: in `2026-04-02-fixed-signflip`, seven of eight chosen submissions came from the trust-region pool;
- five of eight functions were flagged as stagnating in that latest report, which is consistent with the selector shifting toward `trust_region_ts`.

## Assumptions and Limitations

Core assumptions include:

- maximisation rather than minimisation;
- continuous box-constrained inputs already scaled near `[0, 1)`;
- chronological row order as a meaningful history for trust-region reconstruction;
- surrogate smoothness that is at least approximately capturable by the chosen GP kernels and transforms.

Important limitations include:

- **small data per function**: even the largest function has only 49 observations;
- **policy dependence**: later data were chosen by earlier optimisers, so performance summaries are not unbiased;
- **rounding effects**: six-decimal floor quantisation can collapse distinct floating-point candidates into the same portal point;
- **transform sensitivity**: sign-flipped Box-Cox helps all-negative objectives, but it also makes transformed-space diagnostics harder to interpret;
- **trust-region reconstruction**: the trust region is inferred from past observations rather than stored as a native optimiser state

## Ethical Considerations

This is not a human-subject model card, so the most relevant ethical issues are transparency, reproducibility, and responsible claims.

This repository stores candidate-level tables, named recommendation strategies, and seed-level summaries.

The problem is sequential, small-sample, and policy-dependent.

## Why This Level of Detail Is Sufficient

The main modelling and decision choices are already identifiable from the code:

- which transforms are used;
- how candidates are generated;
- how the selector switches under stagnation;
- which diagnostics are logged.

What the repository does **not** provide is an exact command log, git revision, or full per-round provenance. 

The main documentation gaps are:

- no git hash or formal version tag in this workspace copy;
- no exact ten-round manifest;

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
