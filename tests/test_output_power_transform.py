from __future__ import annotations

import importlib.util
import sys
import unittest

import numpy as np


MODULE_PATH = __import__("pathlib").Path(__file__).resolve().parents[1] / "transforms" / "output_power.py"
SPEC = importlib.util.spec_from_file_location("output_power", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
OutputPowerTransform = MODULE.OutputPowerTransform


class OutputPowerTransformTests(unittest.TestCase):
    def test_transform_preserves_maximization_order_for_all_negative_targets(self) -> None:
        y = np.array([-0.4741838512382761, -0.0042843822939058], dtype=float)

        tf = OutputPowerTransform().fit(y)
        y_t = tf.transform(y)

        self.assertTrue(tf.sign_flip_applied)
        self.assertGreater(y[1], y[0])
        self.assertGreater(y_t[1], y_t[0])


if __name__ == "__main__":
    unittest.main()
