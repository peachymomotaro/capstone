#!/usr/bin/env python3
"""Plot observed function-output progress over submission order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import NamedTuple

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/capstonebo_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/capstonebo_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ProgressSummary(NamedTuple):
    function_name: str
    n_submissions: int
    best_index: int
    best_y0: float
    submissions_after_best: int
    best_so_far: pd.Series


def _function_sort_key(path: Path) -> tuple[int, str]:
    try:
        return (int(path.stem.split("_")[-1]), path.stem)
    except ValueError:
        return (10_000, path.stem)


def list_function_csvs(data_dir: Path) -> list[Path]:
    csvs = sorted(Path(data_dir).glob("function_*.csv"), key=_function_sort_key)
    if not csvs:
        raise FileNotFoundError(f"No function CSVs found in {data_dir}")
    return csvs


def load_function_progress(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"function", "index", "y0"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    out = df.loc[:, ["function", "index", "y0"]].copy()
    out["index"] = pd.to_numeric(out["index"], errors="raise").astype(int)
    out["y0"] = pd.to_numeric(out["y0"], errors="raise").astype(float)
    out = out.sort_values("index", kind="mergesort").reset_index(drop=True)
    if out.empty:
        raise ValueError(f"{csv_path} has no observations")
    return out


def summarize_progress(df: pd.DataFrame) -> ProgressSummary:
    work = df.sort_values("index", kind="mergesort").reset_index(drop=True)
    y = work["y0"].astype(float)
    best_pos = int(y.idxmax())
    best_so_far = y.cummax()
    return ProgressSummary(
        function_name=str(work.loc[0, "function"]),
        n_submissions=int(len(work)),
        best_index=int(work.loc[best_pos, "index"]),
        best_y0=float(work.loc[best_pos, "y0"]),
        submissions_after_best=int(len(work) - best_pos - 1),
        best_so_far=best_so_far,
    )


def build_progress_table(data_dir: Path) -> pd.DataFrame:
    rows = []
    for csv_path in list_function_csvs(data_dir):
        summary = summarize_progress(load_function_progress(csv_path))
        rows.append(
            {
                "function": summary.function_name,
                "n_submissions": summary.n_submissions,
                "best_index": summary.best_index,
                "best_y0": summary.best_y0,
                "submissions_after_best": summary.submissions_after_best,
            }
        )
    return pd.DataFrame(rows)


def _annotate_panel(ax, summary: ProgressSummary, x: np.ndarray, y: np.ndarray) -> None:
    best_mask = x == summary.best_index
    best_y = summary.best_y0
    ax.scatter(
        [summary.best_index],
        [best_y],
        marker="*",
        s=160,
        color="#D62728",
        edgecolor="#111111",
        linewidth=0.7,
        zorder=5,
        label="highest y0",
    )

    if summary.submissions_after_best > 0:
        after_x = x[x > summary.best_index]
        if len(after_x) > 0:
            ax.axvspan(
                float(after_x.min()) - 0.5,
                float(x.max()) + 0.5,
                color="#F2CF5B",
                alpha=0.18,
                lw=0,
                label="after highest y0",
            )

    ypos = best_y if np.any(best_mask) else float(np.max(y))
    ax.annotate(
        f"best y0={summary.best_y0:.4g}\n{submissions_after_best_text(summary)}",
        xy=(summary.best_index, ypos),
        xytext=(8, 10),
        textcoords="offset points",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
    )


def submissions_after_best_text(summary: ProgressSummary) -> str:
    return f"{summary.submissions_after_best} submissions after best"


def _plot_one_function(ax, df: pd.DataFrame, summary: ProgressSummary) -> None:
    x = df["index"].to_numpy(dtype=int)
    y = df["y0"].to_numpy(dtype=float)
    best_so_far = summary.best_so_far.to_numpy(dtype=float)

    ax.plot(x, y, color="#4C78A8", lw=1.1, alpha=0.55)
    ax.scatter(x, y, s=28, color="#4C78A8", edgecolor="#1F2933", linewidth=0.45, alpha=0.88, label="observed y0")
    ax.step(x, best_so_far, where="post", color="#111111", lw=1.9, label="best so far")
    _annotate_panel(ax, summary, x, y)

    ax.set_title(summary.function_name.replace("_", " "), fontsize=11, fontweight="bold")
    ax.set_xlabel("submission index")
    ax.set_ylabel("y0")
    ax.grid(alpha=0.18)
    ax.margins(x=0.03, y=0.12)


def plot_function_progress(data_dir: Path, out_path: Path, summary_path: Path | None = None) -> pd.DataFrame:
    csv_paths = list_function_csvs(data_dir)
    n = len(csv_paths)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 3.15 * nrows), squeeze=False)

    summary_rows = []
    for ax, csv_path in zip(axes.ravel(), csv_paths):
        df = load_function_progress(csv_path)
        summary = summarize_progress(df)
        _plot_one_function(ax, df, summary)
        summary_rows.append(
            {
                "function": summary.function_name,
                "n_submissions": summary.n_submissions,
                "best_index": summary.best_index,
                "best_y0": summary.best_y0,
                "submissions_after_best": summary.submissions_after_best,
            }
        )

    for ax in axes.ravel()[n:]:
        ax.set_axis_off()

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Function Output Progress Over Time", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
    return summary_df


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Create a progress plot for observed function outputs.")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--out-path", type=Path, default=root / "images" / "function_progress_over_time.png")
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=root / "images" / "function_progress_over_time_summary.csv",
        help="Optional CSV path for best-y0 and submissions-after-best summary.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = plot_function_progress(
        data_dir=args.data_dir,
        out_path=args.out_path,
        summary_path=args.summary_path,
    )
    print(summary.to_string(index=False))
    print(f"Wrote progress plot to {args.out_path}")
    if args.summary_path is not None:
        print(f"Wrote progress summary to {args.summary_path}")


if __name__ == "__main__":
    main()
