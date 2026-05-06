from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def make_weekly_dashboard(candidates: pd.DataFrame, recommendations: pd.DataFrame, outpath: Path) -> None:
    c = candidates.copy()
    r = recommendations.copy()

    c_sorted = c.sort_values(["function", "is_submission", "pareto_rank", "mu_t"], ascending=[True, False, True, False])

    # Show compact columns for readability.
    show_cols = [
        "function",
        "candidate_label",
        "is_submission",
        "portal_string",
        "mu_t",
        "sigma_t",
        "ei",
        "pi",
        "ucb",
        "is_pareto",
        "pareto_rank",
        "novelty_nn_dist",
        "boundary_prox",
        "duplicate_after_rounding",
    ]
    show_cols = [c for c in show_cols if c in c_sorted.columns]
    c_sorted = c_sorted[show_cols]

    html = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CapstoneBO Weekly Dashboard</title>",
        "<style>",
        "body{font-family:Arial,sans-serif;background:#f7f9fc;color:#1f2937;padding:16px}",
        ".card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:14px;margin-bottom:14px}",
        "table{border-collapse:collapse;width:100%;font-size:12px}",
        "th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:right;white-space:nowrap}",
        "th{text-align:left;background:#f3f4f6}",
        "td:first-child,td:nth-child(2),td:nth-child(4){text-align:left}",
        "</style></head><body>",
        "<div class='card'><h2>Recommendations</h2>",
        r.to_html(index=False, border=0, escape=True),
        "</div>",
        "<div class='card'><h2>Candidate Summary</h2>",
        c_sorted.to_html(index=False, border=0, escape=True),
        "</div>",
        "</body></html>",
    ]

    outpath.write_text("\n".join(html), encoding="utf-8")
