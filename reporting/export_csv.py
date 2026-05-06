from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from reporting.dashboard import make_weekly_dashboard


def write_report_csvs(
    *,
    report_dir: Path,
    candidate_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    portal_strings: pd.DataFrame,
    diagnostics_by_function: dict[str, dict],
    seed_suggestions: pd.DataFrame | None = None,
    stability_summary: pd.DataFrame | None = None,
    run_health: dict | None = None,
    write_dashboard: bool = True,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    candidate_summary.to_csv(report_dir / "candidate_summary.csv", index=False)
    recommendations.to_csv(report_dir / "recommendations.csv", index=False)
    portal_strings.to_csv(report_dir / "portal_strings.csv", index=False)
    if seed_suggestions is not None:
        seed_suggestions.to_csv(report_dir / "seed_suggestions.csv", index=False)
    if stability_summary is not None:
        stability_summary.to_csv(report_dir / "stability_summary.csv", index=False)
    if run_health is not None:
        (report_dir / "run_health.json").write_text(json.dumps(run_health, indent=2), encoding="utf-8")

    # Useful compatibility files for easier diffing against old workflow.
    shortlist = candidate_summary.sort_values(["function", "submission_score"], ascending=[True, False])
    shortlist.to_csv(report_dir / "candidate_shortlist.csv", index=False)

    pivot_cols = [c for c in ["portal_string", "mu_t", "sigma_t", "ei", "pi", "ucb", "submission_score"] if c in shortlist.columns]
    if pivot_cols:
        pivot = shortlist.pivot_table(
            index="function",
            columns="candidate_label",
            values=pivot_cols,
            aggfunc="first",
        )
        pivot.columns = [f"{a}__{b}" for a, b in pivot.columns.to_flat_index()]
        pivot = pivot.reset_index()
        pivot.to_csv(report_dir / "candidate_pivot.csv", index=False)

    for func, payload in diagnostics_by_function.items():
        fdir = report_dir / func
        fdir.mkdir(parents=True, exist_ok=True)
        (fdir / "model_diagnostics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if write_dashboard:
        make_weekly_dashboard(candidate_summary, recommendations, report_dir / "weekly_dashboard.html")
