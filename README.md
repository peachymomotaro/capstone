# CapstoneBO+

`CapstoneBO+` is a repository for a capstone black-box optimisation workflow. It takes cumulative observations for a set of hidden objective functions, fits one surrogate model per function, proposes one new query per function, and writes report artifacts that make the recommendation process inspectable.

The repository is set up for a sequential weekly workflow rather than a one-off benchmark run. The raw data in [`data/`](data) are the running history for eight functions, and the code in this folder rebuilds candidate rankings, recommendations, diagnostics, and portal-formatted outputs from that history.

## Repository At A Glance

- [`data/`](data): cumulative per-function observation histories (`function_1.csv` to `function_8.csv`)
- [`reports/`](reports): dated output folders containing candidate summaries, recommendations, portal strings, and diagnostics
- [`docs/`](docs): project documentation, including the dataset datasheet and optimisation model card
- [`weekly_pack.py`](weekly_pack.py): main orchestration script for rebuilding a weekly run
- [`propose_next.py`](propose_next.py): lightweight entrypoint that prints one proposed portal string per function
- [`EXPLAINER.md`](EXPLAINER.md), [`PIPELINE_WALKTHROUGH.md`](PIPELINE_WALKTHROUGH.md), [`Handover.md`](Handover.md): longer internal project notes and implementation guidance

## How It Works

At a high level, the pipeline:

1. loads the current history for each function from `data/`
2. fits a Gaussian-process-based surrogate model with repository-specific transforms and robustness fallbacks
3. generates a candidate pool from global, trust-region, and elite-region proposals
4. scores candidates with multiple acquisition-style signals
5. removes portal-rounding duplicates and previously observed rounded points
6. chooses one final submission candidate per function and writes report outputs

The default workflow also evaluates multiple random seeds and uses a consensus rule to choose the final portal string written to the report folder.

## Documentation

- [BBO Capstone Dataset Datasheet](docs/datasheet_bbo_capstone_dataset.md)
- [BBO Optimisation Approach Model Card](docs/model_card_bbo_optimisation_approach.md)

## Typical Usage

Run the full weekly rebuild and write a dated report folder:

```bash
python3 weekly_pack.py --report_date 2026-04-02
```

Print one proposed portal input per function without writing a full report:

```bash
python3 propose_next.py
```

If you are running from a package-aware environment, the module form also works:

```bash
python3 -m CapstoneBO+.weekly_pack --report_date 2026-04-02
```

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
