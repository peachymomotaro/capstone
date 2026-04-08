# Datasheet for the CapstoneBO+ Black-Box Optimisation Dataset

## Title and Summary

**Dataset name.** CapstoneBO+ black-box optimisation dataset.

This dataset is the cumulative observation history used by the `CapstoneBO+` repository for a capstone black-box optimisation project. It consists of eight per-function CSV files in [`data/`](../data), one for each hidden objective function, plus dated derived report folders in [`reports/`](../reports) that combine those observations with model diagnostics, candidate rankings, recommendation tables, portal-formatted submissions, and stability summaries. The raw tables are not a conventional supervised-learning benchmark: they record a sequential optimisation process in which historical evaluations are repeatedly reused to propose one new query per function under portal-formatting and duplicate-avoidance constraints.

This datasheet is grounded in the repository’s live code and text materials, especially [`data_io.py`](../data_io.py), [`config.py`](../config.py), [`weekly_pack.py`](../weekly_pack.py), [`EXPLAINER.md`](../EXPLAINER.md), [`PIPELINE_WALKTHROUGH.md`](../PIPELINE_WALKTHROUGH.md), [`Recommendations.md`](../Recommendations.md), [`weekly_tables.ipynb`](../weekly_tables.ipynb), and the dated report outputs under [`reports/`](../reports). No exact file titled “Required capstone component 21.1” was found in the accessible repository files; the closest reflection-style materials are the markdown handover, explainer, walkthrough, and recommendation documents.

## Motivation

The dataset exists to support a capstone exercise in sequential black-box optimisation. The repository structure and code indicate the intended task: given historical evaluations of unknown functions, fit a surrogate model, generate candidate points, choose one portal-safe submission per function, and append later observations as the capstone progresses. The code assumes maximisation, continuous inputs already scaled to approximately `[0, 1)`, and a weekly single-query workflow.

The repository clearly represents a capstone project, but it does not include a formal author list, contributor table, or institution-specific authorship statement. It therefore supports only a limited claim: the dataset appears to have been created and maintained by the capstone project owner/maintainer, but exact creator identities are not documented inside the project files. No funding statement, sponsor acknowledgement, grant number, or external support information was found. No ethics-approval or consent language was found either, which is unsurprising because the data are numerical optimisation traces rather than human-subject records.

An important project assumption, visible throughout the codebase, is that robustness matters as much as nominal prediction quality. The repository repeatedly encodes this through output transforms, input warping, fit fallbacks, trust-region reconstruction, rounding-aware deduplication, and multi-seed consensus reporting. Those choices shape both what data are collected and how those data should be interpreted.

## Composition

The raw observation layer consists of eight CSV files:

| Function | Used dimensions | Rows in current CSV | Unused `x*` columns stored as all-`NaN` | Observed `y0` range |
| --- | ---: | ---: | --- | --- |
| `function_1` | 2 | 19 | `x2`-`x7` | `-0.003606` to `6.83e-12` |
| `function_2` | 2 | 19 | `x2`-`x7` | `-0.065624` to `0.651454` |
| `function_3` | 3 | 24 | `x3`-`x7` | `-0.474184` to `-0.004284` |
| `function_4` | 4 | 39 | `x4`-`x7` | `-32.625660` to `0.608628` |
| `function_5` | 4 | 29 | `x4`-`x7` | `0.112940` to `6049.023450` |
| `function_6` | 5 | 29 | `x5`-`x7` | `-3.172478` to `-0.162953` |
| `function_7` | 6 | 39 | `x6`-`x7` | `0.002701` to `2.244001` |
| `function_8` | 8 | 49 | none | `5.592193` to `9.920822` |

Across those files there are 247 observations in total. Every raw CSV uses the same schema: `function`, `index`, `x0`-`x7`, `y0`. The repository’s loader in [`data_io.py`](../data_io.py) treats populated `x*` columns as active dimensions and ignores columns that are entirely missing. In the current files:

- there are no missing values in `y0`;
- there are no missing values in active input dimensions;
- the only `NaN` values are placeholder entries for unused higher-numbered coordinates.

This means the dataset is heterogeneous in dimension but homogeneous in file format. That design simplifies code reuse at the cost of storing many explicit placeholders.

The raw CSVs are not the only dataset objects in the repository. The project also maintains combined, derived tables under dated report folders such as [`reports/2026-04-02-fixed-signflip/`](../reports/2026-04-02-fixed-signflip). These include:

- `candidate_summary.csv`: pooled candidate-level scores and diagnostics across all functions;
- `recommendations.csv`: named strategy recommendations such as `chosen_submission`, `primary_pareto_sigma`, `trust_region_ts`, and `global_explore`;
- `portal_strings.csv`: final portal-formatted suggestions;
- `seed_suggestions.csv` and `stability_summary.csv`: multi-seed robustness summaries;
- `run_health.json` and per-function `model_diagnostics.json`: run metadata and modelling diagnostics.

The dataset should therefore be understood as a layered artefact: raw evaluation history plus generated weekly analysis outputs.

Standard train/validation/test splits are not especially appropriate here. The sequential objective is not to predict held-out labels under i.i.d. assumptions, but to choose the next evaluation point under a query budget. More appropriate evaluation logic is chronological and decision-focused: best-so-far trajectories, improvement over time, recommendation stability, or retrospective comparison of strategies on the evolving history.

Relationships between records are strong. Rows are grouped by function, appear to be ordered chronologically, and are not statistically independent because later observations are chosen in response to earlier ones. Functions are independent tasks sharing a common schema, not repeated measurements of the same phenomenon.

Privacy and sensitive-content risk are low. The data contain numerical coordinates and objective values only. No people, organisations, or identifiable entities are represented in the raw records.

## Collection Process

The repository suggests a weekly closed-loop collection process rather than a one-time batch export. The notebook title in [`weekly_tables.ipynb`](../weekly_tables.ipynb) refers to “after you append the latest weekly results,” and the main orchestration code in [`weekly_pack.py`](../weekly_pack.py) is explicitly built to produce one new portal-ready point per function from the currently available history.

Based on the live `CapstoneBO+` code, query generation works as follows:

1. Load the current history for one function.
2. Fit a Gaussian-process surrogate with optional output transform and input warp.
3. Reconstruct a trust region from the historical sequence.
4. Generate a mixed candidate pool:
   - a global Sobol sample over the full box;
   - a trust-region Sobol sample around the current incumbent;
   - an “elite” Sobol sample inside a box around top observed points.
5. Score candidates with EI, logEI, PI, a sweep of UCB values, Thompson-style scores, and an exploration score based on uncertainty times novelty.
6. Remove duplicates after six-decimal floor rounding, including collisions with already observed points.
7. Choose a final suggestion using a portfolio rule that can switch between a Pareto-plus-sigma selector, trust-region Thompson-style selection, and explicit global exploration.
8. In the default workflow, repeat across seeds `0`-`4` and report a consensus portal string.

This process is mixed rather than purely deterministic. It combines deterministic data cleaning and rounding rules with stochastic Sobol candidate generation, randomised Thompson-style scores, and multi-seed consensus.

The dated report folders provide partial evidence for timing and incremental accumulation:

- `2026-03-19-plus` stores an early `CapstoneBO+` report snapshot;
- `2026-03-20` stores a later snapshot with one additional observation per function relative to `2026-03-19-plus`;
- `2026-04-02` stores another snapshot, and the sibling folders `2026-04-02-audited` and `2026-04-02-fixed-signflip` match the current raw-data row counts.

Those dates support a sequential cadence, but the repository does not include a standalone round manifest that maps every observation to an explicit “round 1” through “round 10” label. It would therefore be unsafe to claim an exact round-by-round reconstruction beyond what the dated files directly show.

No ethical-review or consent process is documented. For this dataset, that is best treated as “not applicable and not discussed,” not as evidence of formal review.

The collection process has several implications for downstream conclusions:

- the data are policy-dependent, because the optimiser helped choose later points;
- the sample is not representative of the full search space;
- apparent improvements may reflect both better optimisation and cumulative luck;
- comparisons across functions are scale-sensitive because the raw objective ranges differ greatly.

## Preprocessing, Cleaning, and Labelling

The raw CSVs appear close to the stored optimisation history, but the repository adds several interpretation-relevant preprocessing steps at analysis time.

At load time, [`data_io.py`](../data_io.py):

- identifies active dimensions by checking which `x*` columns contain any non-missing values;
- requires the presence of `y0`;
- drops rows with missing values in active inputs or `y0`.

In the current raw files, those drop rules do not remove any rows, because active inputs and `y0` are complete.

For modelling, the code then applies transformations that do **not** change the stored CSVs but do change how values are interpreted:

- the default output transform in [`transforms/output_power.py`](../transforms/output_power.py) uses Box-Cox when all targets are strictly positive, Box-Cox on sign-flipped targets when all targets are strictly negative, and Yeo-Johnson otherwise;
- an optional copula-style output transform exists in [`transforms/output_copula.py`](../transforms/output_copula.py), but no stored report examined here used it as the selected transform;
- inputs can be warped with BoTorch’s `Warp` or a Kumaraswamy fallback before GP fitting;
- candidate-level diagnostics add novelty ratios, boundary penalties, transformed-space posterior summaries, and rounded-coordinate columns such as `x0_q`.

The current report diagnostics show sign-flipped Box-Cox behaviour for all-negative functions such as `function_3` and `function_6`, and an explicit no-transform override for `function_1`.

This dataset has no class labels or semantic annotations in the supervised-learning sense. The repository’s “labels” are operational metadata such as function identifier, candidate origin (`global`, `tr`, `elite`), recommendation type, and model diagnostics.

## Uses

Intended uses include:

- retrospective analysis of the capstone optimisation process;
- rerunning or auditing the `CapstoneBO+` weekly pipeline;
- comparing candidate-generation and selection heuristics on the same cumulative histories;
- producing portal-ready weekly suggestions for the eight repository functions.

Appropriate use is narrow and contextual. This dataset supports claims about this capstone’s sequential optimisation workflow far better than claims about generic black-box optimisation performance in the abstract.

Inappropriate uses include:

- treating the data as an ordinary i.i.d. regression benchmark with random splits;
- making broad claims about all Bayesian optimisation methods from this single capstone setting;
- using transformed-space report quantities such as `mu_t` as if they were directly comparable raw objective values across functions;
- extrapolating from these functions to constrained, categorical, multi-fidelity, or batch settings not represented in the stored code and data.

The main risks and failure modes are methodological rather than social:

- small sample sizes for some functions;
- strong closed-loop sampling bias;
- substantial scale heterogeneity across objectives;
- sensitivity to rounding, trust-region reconstruction, and stochastic candidate generation;
- report variants (`2026-04-02`, `-audited`, `-fixed-signflip`) that can drift out of sync with the raw data if not clearly documented.

Fairness language aimed at human populations is not especially relevant here. The more relevant responsibility issue is transparent reporting of what the data do and do not support.

## Distribution

The dataset lives inside this repository copy:

- raw observation history in [`data/`](../data);
- combined weekly outputs in [`reports/`](../reports);
- quick tabulation notebook in [`weekly_tables.ipynb`](../weekly_tables.ipynb);
- related figures in [`images/`](../images).

No separate release channel, external mirror, or public data package is documented in the repository. No dataset-specific licence file was found in `CapstoneBO+`, and there is no explicit usage-terms statement. The safest interpretation is therefore that licence and redistribution terms are undocumented in this repository copy.

The raw data files themselves have no external dependencies, but reproducing the derived outputs requires the repository’s Python stack, including pandas, NumPy, PyTorch, BoTorch, GPyTorch, and scikit-learn.

## Maintenance

The repository appears to be maintained in a cumulative, working-project style rather than through formal data releases. In practice, versioning occurs through:

- appending new observations to the per-function CSV files;
- generating dated report folders;
- keeping alternative reruns such as `2026-04-02-audited` and `2026-04-02-fixed-signflip`.

Those dated folders are useful archival artefacts, but they also introduce a maintenance requirement: when raw CSVs are updated, previously generated reports may no longer match the current raw data. That mismatch is already visible here, because the current raw row counts align with the `2026-04-02-audited` and `2026-04-02-fixed-signflip` runs rather than the plain `2026-04-02` folder.

The repository does not expose a formal data steward, issue tracker for the dataset, or release checklist. Future updates would ideally include:

- explicit per-round provenance for appended observations;
- a statement of which dated report folder is authoritative;
- an explicit licence;
- a short author/maintainer statement.

## Known Limitations and Documentation Gaps

- No exact reflection file named “Required capstone component 21.1” was found in the accessible repository files.
- The repository does not name the dataset creators formally.
- No funding, licence, or redistribution terms are documented.
- The raw files preserve chronological accumulation, but not an explicit round manifest.
- The latest raw data match the `2026-04-02-audited` and `2026-04-02-fixed-signflip` reports, not the plain `2026-04-02` folder.
- The dataset is strongly policy-dependent because later rows were selected by earlier versions of the optimiser.
