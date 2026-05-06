from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "plots" / "plot_highdim_function_summaries.py"
SPEC = importlib.util.spec_from_file_location("plot_highdim_function_summaries", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlotHighdimFunctionSummariesTests(unittest.TestCase):
    def test_panel_tsne_scatter_uses_random_init(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.random((12, 4))
        y = np.linspace(0.0, 1.0, 12)
        fig, ax = plt.subplots()
        try:
            with patch.object(MODULE, "TSNE") as mock_tsne:
                instance = mock_tsne.return_value
                instance.fit_transform.return_value = np.column_stack((y, y))
                MODULE._panel_tsne_scatter(ax, X, y, cmap="viridis", random_state=7)
            self.assertEqual(mock_tsne.call_args.kwargs["init"], "random")
        finally:
            plt.close(fig)

    def test_panel_tsne_scatter_returns_scatter_and_title(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.random((12, 4))
        y = np.linspace(0.0, 1.0, 12)
        fig, ax = plt.subplots()
        try:
            sc = MODULE._panel_tsne_scatter(ax, X, y, cmap="viridis", random_state=7)
            self.assertIsNotNone(sc)
            self.assertIn("t-SNE", ax.get_title())
            offsets = sc.get_offsets()
            self.assertEqual(offsets.shape, (12, 2))
        finally:
            plt.close(fig)

    def test_plot_function_summary_writes_image_for_highdim_function(self) -> None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "function_8.csv"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "function_8_summary.png"
            MODULE.plot_function_summary(
                csv_path=csv_path,
                out_path=out_path,
                grid_size=20,
                top_features=2,
                seed=0,
            )
            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
