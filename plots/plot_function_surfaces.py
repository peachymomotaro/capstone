#!/usr/bin/env python3
"""Create 3-panel surface visualizations for function_1 and function_2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/capstonebo_mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/capstonebo_cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay, QhullError


def _load_function(csv_path: Path, function_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    sub = df[df["function"] == function_name].copy()
    for col in ("x0", "x1", "y0"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.dropna(subset=["x0", "x1", "y0"]).reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"No usable x0/x1/y0 rows for {function_name} in {csv_path}")
    return sub


def _fit_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    grid_size: int = 140,
    kernel: str = "thin_plate_spline",
    smoothing: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.column_stack((x, y))
    model = RBFInterpolator(points, z, kernel=kernel, smoothing=smoothing)

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    x_pad = 0.08 * (x_max - x_min) if x_max > x_min else 0.05
    y_pad = 0.08 * (y_max - y_min) if y_max > y_min else 0.05

    grid_x = np.linspace(max(0.0, x_min - x_pad), min(1.0, x_max + x_pad), grid_size)
    grid_y = np.linspace(max(0.0, y_min - y_pad), min(1.0, y_max + y_pad), grid_size)
    xx, yy = np.meshgrid(grid_x, grid_y)

    grid_points = np.column_stack((xx.ravel(), yy.ravel()))
    zz = model(grid_points).reshape(xx.shape)

    # Keep contour/surface inside the convex hull for a cleaner shape.
    try:
        hull = Delaunay(points)
        inside = hull.find_simplex(grid_points) >= 0
        mask = inside.reshape(xx.shape)
        zz = np.where(mask, zz, np.nan)
    except QhullError:
        pass

    return xx, yy, zz


def _pretty_name(function_name: str) -> str:
    return function_name.replace("_", " ").title()


def plot_triptych(sub: pd.DataFrame, function_name: str, out_path: Path) -> None:
    x = sub["x0"].to_numpy(dtype=float)
    y = sub["x1"].to_numpy(dtype=float)
    z = sub["y0"].to_numpy(dtype=float)
    xx, yy, zz = _fit_surface(x, y, z)

    cmap = "viridis"
    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    fig.patch.set_facecolor("#f2f2f2")
    fig.suptitle(
        f"{_pretty_name(function_name)} - 3D (x0, x1, y)",
        fontsize=17,
        fontweight="bold",
    )

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    sc1 = ax1.scatter(
        x,
        y,
        z,
        c=z,
        cmap=cmap,
        s=175,
        edgecolor="#222222",
        linewidth=1.7,
        alpha=0.98,
    )
    ax1.set_title("3D Scatter", pad=15, fontsize=13, fontweight="bold")
    ax1.set_xlabel("x0")
    ax1.set_ylabel("x1")
    ax1.set_zlabel("y (target)")
    ax1.view_init(elev=22, azim=50)
    cbar1 = fig.colorbar(sc1, ax=ax1, shrink=0.62, pad=0.06)
    cbar1.set_label("y")

    ax2 = fig.add_subplot(1, 3, 2)
    levels = 24
    cont = ax2.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.98)
    ax2.contour(xx, yy, zz, levels=levels, colors="k", linewidths=0.25, alpha=0.18)
    ax2.scatter(
        x,
        y,
        c=z,
        cmap=cmap,
        s=175,
        edgecolor="#222222",
        linewidth=1.6,
        zorder=3,
    )
    ax2.set_title("Contour / Input-Output", fontsize=13, fontweight="bold")
    ax2.set_xlabel("x0", fontweight="bold")
    ax2.set_ylabel("x1", fontweight="bold")
    ax2.grid(alpha=0.15)
    cbar2 = fig.colorbar(cont, ax=ax2, shrink=0.98, pad=0.04)
    cbar2.set_label("y")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    surf = ax3.plot_surface(xx, yy, zz, cmap=cmap, linewidth=0, antialiased=True, alpha=0.98)
    ax3.scatter(
        x,
        y,
        z,
        s=125,
        c="#ff4b2b",
        edgecolor="#5a1200",
        linewidth=1.3,
        depthshade=True,
    )
    ax3.set_title("Fitted Surface", pad=15, fontsize=13, fontweight="bold")
    ax3.set_xlabel("x0")
    ax3.set_ylabel("x1")
    ax3.set_zlabel("y (target)")
    ax3.view_init(elev=20, azim=50)
    cbar3 = fig.colorbar(surf, ax=ax3, shrink=0.62, pad=0.06)
    cbar3.set_label("y")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Plot Function 1/2 3-panel visualizations.")
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--out-dir", type=Path, default=root / "images")
    parser.add_argument(
        "--functions",
        nargs="+",
        default=["function_1", "function_2"],
        help="Function names that map to <function>.csv in --data-dir",
    )
    args = parser.parse_args()

    for fn_name in args.functions:
        csv_path = args.data_dir / f"{fn_name}.csv"
        sub = _load_function(csv_path, fn_name)
        out_path = args.out_dir / f"{fn_name}_surface_triptych.png"
        plot_triptych(sub, fn_name, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
