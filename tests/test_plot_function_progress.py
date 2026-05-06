from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "plots" / "plot_function_progress.py"
SPEC = importlib.util.spec_from_file_location("plot_function_progress", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PlotFunctionProgressTests(unittest.TestCase):
    def test_progress_summary_counts_submissions_after_highest_y0(self) -> None:
        df = pd.DataFrame(
            {
                "function": ["function_1"] * 5,
                "index": [0, 1, 2, 3, 4],
                "x0": [0.1, 0.2, 0.3, 0.4, 0.5],
                "y0": [1.0, 3.0, 2.5, 2.0, 2.8],
            }
        )

        summary = MODULE.summarize_progress(df)

        self.assertEqual(summary.function_name, "function_1")
        self.assertEqual(summary.best_index, 1)
        self.assertEqual(summary.submissions_after_best, 3)
        self.assertEqual(summary.best_so_far.tolist(), [1.0, 3.0, 3.0, 3.0, 3.0])

    def test_plot_function_progress_writes_image(self) -> None:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "function_progress.png"

            MODULE.plot_function_progress(data_dir=data_dir, out_path=out_path)

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
