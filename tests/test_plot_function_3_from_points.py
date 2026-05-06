from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "plots" / "plot_function_3_from_points.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("plot_function_3_from_points", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PlotFunction3FromPointsTests(unittest.TestCase):
    def test_write_function_3_points_plot_creates_html(self) -> None:
        self.assertTrue(MODULE_PATH.exists(), f"Missing module: {MODULE_PATH}")

        module = _load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "function_3_interactive_from_points.html"
            module.write_function_3_points_plot(points=module.POINTS, out_path=out_path)

            self.assertTrue(out_path.exists())
            self.assertGreater(out_path.stat().st_size, 0)
            html = out_path.read_text(encoding="utf-8")
            self.assertIn("Function 3 Interactive 3D Scatter", html)
            self.assertIn("x0", html)
            self.assertIn("y0", html)


if __name__ == "__main__":
    unittest.main()
