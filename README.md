# CapstoneBO+

`CapstoneBO+` is a repository for a capstone black-box optimisation workflow. 

The model is a Bayesian optimisation pipeline. For each function, it fits a Gaussian process surrogate to the observed inputs and outputs. The preferred model uses an additive linear plus Matern kernel, optional input warping, and output transformations such as Box-Cox or Yeo-Johnson when useful. The pipeline then generates candidate points from global Sobol samples, trust-region samples, and elite-region samples around historically strong areas.

Candidates are scored with acquisition-style signals including expected improvement, probability of improvement, upper confidence bound, Thompson-style scores, and an exploration score based on uncertainty and novelty. A portfolio selector chooses the final weekly recommendation.

The repository is set up for a sequential weekly workflow. The raw data in [`data/`](data) are the running history for eight functions, and the code in this folder rebuilds candidate rankings, recommendations, diagnostics, and portal-formatted outputs from that history.

## Contact Details

Prepared for a machine learning capstone submission.

Get in touch via email: curry.peter@googlemail.com
My website: https://petercurry.org/

## Start Here

- **Model card:** [`docs/Model_Card.md`](docs/Model_Card.md) explains the optimisation method, modelling choices, performance summary, assumptions, and limitations.
- **Dataset datasheet:** [`docs/Datasheet.md`](docs/Datasheet.md) documents the data files, columns, collection process, and known caveats.

## Repository At A Glance

- [`data/`](data): cumulative per-function observation histories (`function_1.csv` to `function_8.csv`)
- [`reports/`](reports): dated output folders containing candidate summaries, recommendations, portal strings, and diagnostics
- [`docs/`](docs): project documentation, especially the [`model card`](docs/Model_Card.md) and [`dataset datasheet`](docs/Datasheet.md)
- [`plots/`](plots): plotting and visualization scripts for progress charts, surfaces, and high-dimensional summaries
- [`notebooks/`](notebooks): supporting exploratory notebooks
- [`weekly_pack.py`](weekly_pack.py): main orchestration script for rebuilding a weekly run
- [`propose_next.py`](propose_next.py): lightweight entrypoint that prints one proposed portal string per function

## How It Works

At a high level, the pipeline:

1. loads the current history for each function from `data/`
2. fits a Gaussian-process-based surrogate model with repository-specific transforms and robustness fallbacks
3. generates a candidate pool from global, trust-region, and elite-region proposals
4. scores candidates with multiple acquisition-style signals
5. removes portal-rounding duplicates and previously observed rounded points
6. chooses one final submission candidate per function and writes report outputs

The default workflow also evaluates multiple random seeds and uses a consensus rule to choose the final portal string written to the report folder.

## Typical Usage

Run the full weekly rebuild and write a dated report folder:

`python3 weekly_pack.py --report_date 2026-04-30`

Print one proposed portal input per function without writing a full report:

`python3 propose_next.py`

Regenerate progress and visualization artifacts:

`python3 plots/plot_function_progress.py`

`python3 plots/plot_function_surfaces.py`

## Hyperparameter Optimisation

The GP hyperparameters are fitted automatically during model training. Important configuration choices include:

- kernel mode: `lin_matern32_add`;
- output transform mode: `power`;
- global Sobol candidates: `3072`;
- trust-region candidates: `1024`;
- elite-region candidates: `512`;
- UCB beta sweep: `0.50` to `3.00`;
- robust consensus over five random seeds.

## Report Outputs

Each dated folder in [`reports/`](reports) typically contains:

- `candidate_summary.csv`
- `recommendations.csv`
- `portal_strings.csv`
- `seed_suggestions.csv`
- `stability_summary.csv`
- `run_health.json`
- `weekly_dashboard.html`
- `function_k/model_diagnostics.json`

These outputs are intended to support inspection and auditing, not just final submission.
