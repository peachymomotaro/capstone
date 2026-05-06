#!/usr/bin/env python3
"""Create multi-panel summary visualizations for high-dimensional functions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from config import RebuildConfig
from data_io import load_function_dataset
from models.warped_gp import fit_warped_gp


def _feature_ranking(df: pd.DataFrame, x_cols: list[str], y_col: str = "y0") -> list[tuple[str, float]]:
    pairs: list[tuple[str, float, float]] = []
    y = df[y_col].to_numpy(dtype=float)
    for col in x_cols:
        x = df[col].to_numpy(dtype=float)
        corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
        var = float(np.var(x))
        pairs.append((col, corr, var))

    # Primary rank: abs(corr) descending, NaN last. Secondary rank: variance descending.
    pairs.sort(
        key=lambda t: (
            np.isnan(t[1]),
            -(abs(t[1]) if np.isfinite(t[1]) else -1.0),
            -t[2],
        )
    )
    return [(c, abs(v) if np.isfinite(v) else float("nan")) for c, v, _ in pairs]


def _choose_top_features(df: pd.DataFrame, x_cols: list[str], count: int) -> list[str]:
    count = max(1, int(count))
    ranked = _feature_ranking(df, x_cols)
    finite = [c for c, s in ranked if np.isfinite(s)]
    if len(finite) >= count:
        return finite[:count]

    # Fallback: highest variance features.
    variances = sorted(
        ((c, float(np.var(df[c].to_numpy(dtype=float)))) for c in x_cols),
        key=lambda t: t[1],
        reverse=True,
    )
    return [v[0] for v in variances[:count]]


def _base_grid_from_median(medians: np.ndarray, n: int) -> np.ndarray:
    return np.tile(medians[None, :], (n, 1))


def _predict_mu(model, X_np: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    x_t = torch.tensor(X_np, dtype=dtype)
    with torch.no_grad():
        post = model.posterior(x_t)
        mu = post.mean.detach().cpu().numpy().reshape(-1)
    return mu.astype(float)


def _panel_pca_scatter(ax, X: np.ndarray, y: np.ndarray, cmap: str):
    if X.shape[0] < 2 or X.shape[1] < 2:
        ax.text(0.5, 0.5, "Insufficient data for PCA", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=0)
    pcs = pca.fit_transform(Xs)
    sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=y, cmap=cmap, s=70, edgecolor="#111111", linewidth=0.6)
    evr = pca.explained_variance_ratio_
    ax.set_title(f"PCA Scatter (PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    return pca, sc


def _panel_loadings(ax, pca: PCA | None, x_cols: list[str]):
    if pca is None:
        ax.text(0.5, 0.5, "No PCA loadings available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    load = np.sqrt(np.sum(np.square(pca.components_[:2, :]), axis=0))
    idx = np.argsort(load)[::-1][: min(8, len(load))]
    names = [x_cols[i] for i in idx]
    vals = load[idx]
    ax.barh(np.arange(len(names))[::-1], vals, color="#4C78A8")
    ax.set_yticks(np.arange(len(names))[::-1], labels=names)
    ax.set_title("PCA Loading Strength (PC1+PC2)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Magnitude")
    ax.grid(alpha=0.15, axis="x")


def _safe_tsne_perplexity(n_samples: int) -> float | None:
    if n_samples < 3:
        return None
    return float(min(30, max(2, n_samples // 3), n_samples - 1))


def _panel_tsne_scatter(ax, X: np.ndarray, y: np.ndarray, cmap: str, random_state: int):
    if X.shape[0] < 3:
        ax.text(0.5, 0.5, "Insufficient data for t-SNE", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    perplexity = _safe_tsne_perplexity(X.shape[0])
    if perplexity is None:
        ax.text(0.5, 0.5, "Insufficient data for t-SNE", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return None

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=int(random_state),
        init="random",
        learning_rate="auto",
        max_iter=1000,
    )
    emb = tsne.fit_transform(Xs)
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, s=70, edgecolor="#111111", linewidth=0.6)
    ax.set_title(f"t-SNE Scatter (perplexity={perplexity:.0f})", fontsize=11, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.2)
    return sc


def _panel_pair(ax, df: pd.DataFrame, feat_a: str, feat_b: str, cmap: str):
    sc = ax.scatter(
        df[feat_a].to_numpy(dtype=float),
        df[feat_b].to_numpy(dtype=float),
        c=df["y0"].to_numpy(dtype=float),
        cmap=cmap,
        s=70,
        edgecolor="#111111",
        linewidth=0.6,
    )
    ax.set_title(f"Observed Pair: {feat_a} vs {feat_b}", fontsize=11, fontweight="bold")
    ax.set_xlabel(feat_a)
    ax.set_ylabel(feat_b)
    ax.grid(alpha=0.2)
    return sc


def _panel_gp_surface(
    ax,
    model,
    dtype: torch.dtype,
    x_cols: list[str],
    feat_a: str,
    feat_b: str,
    medians: np.ndarray,
    df: pd.DataFrame,
    grid_size: int,
    cmap: str,
):
    ia = x_cols.index(feat_a)
    ib = x_cols.index(feat_b)

    gx = np.linspace(0.0, 1.0, grid_size)
    gy = np.linspace(0.0, 1.0, grid_size)
    xx, yy = np.meshgrid(gx, gy)
    grid = _base_grid_from_median(medians, xx.size)
    grid[:, ia] = xx.ravel()
    grid[:, ib] = yy.ravel()

    mu = _predict_mu(model=model, X_np=grid, dtype=dtype).reshape(xx.shape)
    cont = ax.contourf(xx, yy, mu, levels=22, cmap=cmap, alpha=0.96)
    ax.contour(xx, yy, mu, levels=14, colors="k", linewidths=0.25, alpha=0.25)
    ax.scatter(
        df[feat_a].to_numpy(dtype=float),
        df[feat_b].to_numpy(dtype=float),
        c=df["y0"].to_numpy(dtype=float),
        cmap=cmap,
        s=42,
        edgecolor="#111111",
        linewidth=0.5,
    )
    ax.set_title("GP Mean Slice (2D)", fontsize=11, fontweight="bold")
    ax.set_xlabel(feat_a)
    ax.set_ylabel(feat_b)
    ax.grid(alpha=0.15)
    return cont


def _panel_1d_pd(
    ax,
    model,
    dtype: torch.dtype,
    x_cols: list[str],
    top_feats: list[str],
    medians: np.ndarray,
):
    colors = ["#1F77B4", "#D62728", "#2CA02C", "#9467BD"]
    xline = np.linspace(0.0, 1.0, 140)
    for i, feat in enumerate(top_feats):
        idx = x_cols.index(feat)
        grid = _base_grid_from_median(medians, xline.size)
        grid[:, idx] = xline
        mu = _predict_mu(model=model, X_np=grid, dtype=dtype)
        ax.plot(xline, mu, lw=2.2, color=colors[i % len(colors)], label=feat)
    ax.set_title("1D Partial Dependence (GP Mean)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Predicted mean")
    ax.set_xlim(0.0, 1.0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, loc="best", fontsize=9)


def _panel_target_stats(
    ax,
    y: np.ndarray,
    function_name: str,
    x_cols: list[str],
    top_feats: list[str],
    ranking: list[tuple[str, float]],
):
    ax.hist(y, bins=min(12, max(6, len(y) // 2)), color="#72B7B2", alpha=0.85, edgecolor="#0E4B50")
    ax.set_title("Target Distribution + Stats", fontsize=11, fontweight="bold")
    ax.set_xlabel("y0")
    ax.set_ylabel("count")
    lines = [
        f"{function_name}",
        f"n={len(y)}  d={len(x_cols)}",
        f"min={np.min(y):.4g}  max={np.max(y):.4g}",
        f"median={np.median(y):.4g}",
        f"top features: {', '.join(top_feats)}",
    ]
    if ranking:
        top_rank = " | ".join(
            [f"{c}:{(s if np.isfinite(s) else float('nan')):.3f}" for c, s in ranking[:4]]
        )
        lines.append(f"|corr(x,y0)|: {top_rank}")
    ax.text(
        0.98,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )


def _render_no_gp(axs, reason: str):
    axs[1, 0].text(0.5, 0.5, f"GP panel unavailable\n{reason}", ha="center", va="center", transform=axs[1, 0].transAxes)
    axs[1, 0].set_axis_off()
    axs[1, 1].text(0.5, 0.5, f"1D PD unavailable\n{reason}", ha="center", va="center", transform=axs[1, 1].transAxes)
    axs[1, 1].set_axis_off()


def plot_function_summary(csv_path: Path, out_path: Path, grid_size: int, top_features: int, seed: int) -> None:
    if top_features < 2:
        raise ValueError("--top-features must be >= 2")

    ds = load_function_dataset(csv_path, dtype=torch.double)
    if ds.n < 8:
        raise ValueError(f"{ds.function_id}: need at least 8 usable rows, got {ds.n}")

    df = ds.df_raw.copy().reset_index(drop=True)
    x_cols = list(ds.x_cols)
    y_np = ds.y.detach().cpu().numpy().reshape(-1).astype(float)
    X_np = ds.X.detach().cpu().numpy().astype(float)
    ranking = _feature_ranking(df, x_cols)
    top_feats = _choose_top_features(df, x_cols, count=max(2, top_features))
    if len(top_feats) < 2:
        top_feats = list(x_cols[:2])
    top2 = top_feats[:2]
    medians = np.median(X_np, axis=0)

    cmap = "viridis"
    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.patch.set_facecolor("#f4f6f9")
    fig.suptitle(
        f"{ds.function_id.replace('_', ' ').title()} Summary (n={ds.n}, d={ds.d})",
        fontsize=16,
        fontweight="bold",
    )

    pca_model, sc_pca = _panel_pca_scatter(axs[0, 0], X_np, y_np, cmap=cmap)
    _panel_loadings(axs[0, 1], pca_model, x_cols=x_cols)
    sc_pair = _panel_tsne_scatter(axs[0, 2], X_np, y_np, cmap=cmap, random_state=int(seed))
    if sc_pair is None:
        sc_pair = sc_pca
        axs[0, 2].text(
            0.5,
            0.5,
            "t-SNE panel unavailable",
            ha="center",
            va="center",
            transform=axs[0, 2].transAxes,
        )
        axs[0, 2].set_axis_off()

    gp_error = None
    model = None
    try:
        cfg = RebuildConfig(seed=int(seed))
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        fitted = fit_warped_gp(ds.X, ds.y, cfg)
        model = fitted.model
    except Exception as exc:  # pragma: no cover - used for robust fallback in user runs.
        gp_error = str(exc)

    if model is not None and len(x_cols) >= 2:
        try:
            cont = _panel_gp_surface(
                axs[1, 0],
                model=model,
                dtype=ds.X.dtype,
                x_cols=x_cols,
                feat_a=top2[0],
                feat_b=top2[1],
                medians=medians,
                df=df,
                grid_size=int(grid_size),
                cmap=cmap,
            )
            _panel_1d_pd(
                axs[1, 1],
                model=model,
                dtype=ds.X.dtype,
                x_cols=x_cols,
                top_feats=top_feats,
                medians=medians,
            )
            cbar_gp = fig.colorbar(cont, ax=axs[1, 0], shrink=0.9, pad=0.02)
            cbar_gp.set_label("predicted mean (transformed target)", fontsize=9)
        except Exception as exc:  # pragma: no cover - robust fallback.
            gp_error = str(exc)
            _render_no_gp(axs, reason=gp_error)
    elif model is None:
        _render_no_gp(axs, reason=gp_error or "unknown model fitting error")
    else:
        _render_no_gp(axs, reason="d < 2: 2D GP slice requires at least two features")

    _panel_target_stats(
        axs[1, 2],
        y=y_np,
        function_name=ds.function_id,
        x_cols=x_cols,
        top_feats=top_feats,
        ranking=ranking,
    )

    cbar_top = fig.colorbar(sc_pca, ax=axs[0, 0], shrink=0.85, pad=0.02)
    cbar_top.set_label("y0", fontsize=9)
    cbar_pair = fig.colorbar(sc_pair, ax=axs[0, 2], shrink=0.85, pad=0.02)
    cbar_pair.set_label("y0", fontsize=9)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Plot high-dimensional function summary panels for CapstoneBO.")
    ap.add_argument("--data-dir", type=Path, default=root / "CapstoneBO" / "data")
    ap.add_argument("--out-dir", type=Path, default=root / "CapstoneBO" / "images")
    ap.add_argument(
        "--functions",
        nargs="+",
        default=[f"function_{k}" for k in range(3, 9)],
        help="Function ids that map to <function>.csv in --data-dir",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-size", type=int, default=120)
    ap.add_argument("--top-features", type=int, default=2)
    return ap


def main() -> None:
    args = _build_parser().parse_args()
    for fn in args.functions:
        csv_path = args.data_dir / f"{fn}.csv"
        out_path = args.out_dir / f"{fn}_summary.png"
        plot_function_summary(
            csv_path=csv_path,
            out_path=out_path,
            grid_size=int(args.grid_size),
            top_features=int(args.top_features),
            seed=int(args.seed),
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
