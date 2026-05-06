#!/usr/bin/env python3
"""Create an interactive 3D scatter plot for function_3."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


FUNCTION_NAME = "function_3"


def _load_function_3(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sub = df[df["function"] == FUNCTION_NAME].copy()
    for col in ("x0", "x1", "x2", "y0"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=["x0", "x1", "x2", "y0"]).reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No usable x0/x1/x2/y0 rows for {FUNCTION_NAME} in {csv_path}")
    return sub


def build_function_3_interactive_figure(csv_path: Path) -> go.Figure:
    sub = _load_function_3(csv_path)
    customdata = sub[["index", "y0"]].to_numpy()

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=sub["x0"],
                y=sub["x1"],
                z=sub["x2"],
                mode="markers",
                customdata=customdata,
                marker={
                    "size": 7,
                    "color": sub["y0"],
                    "colorscale": "Viridis",
                    "opacity": 0.9,
                    "line": {"color": "#1a1a1a", "width": 1},
                    "colorbar": {"title": "y0"},
                },
                hovertemplate=(
                    "index=%{customdata[0]}<br>"
                    "x0=%{x:.6f}<br>"
                    "x1=%{y:.6f}<br>"
                    "x2=%{z:.6f}<br>"
                    "y0=%{customdata[1]:.6f}<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title="Function 3 Interactive 3D Scatter",
        scene={
            "xaxis_title": "x0",
            "yaxis_title": "x1",
            "zaxis_title": "x2",
        },
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
    )
    return fig


def write_function_3_interactive_plot(csv_path: Path, out_path: Path) -> None:
    fig = build_function_3_interactive_figure(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), full_html=True, include_plotlyjs=True)


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Create an interactive 3D plot for function_3.")
    parser.add_argument("--csv-path", type=Path, default=root / "data" / "function_3.csv")
    parser.add_argument(
        "--out-path",
        type=Path,
        default=root / "images" / "function_3_interactive_3d.html",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    write_function_3_interactive_plot(csv_path=args.csv_path, out_path=args.out_path)
    print(f"Saved: {args.out_path}")


if __name__ == "__main__":
    main()
