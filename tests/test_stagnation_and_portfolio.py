from __future__ import annotations

import sys
import unittest
from contextlib import contextmanager
from os import chdir
from pathlib import Path

import numpy as np
import torch


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import weekly_pack
from candidates.selectors import select_portfolio
from config import RebuildConfig


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    chdir(path)
    try:
        yield
    finally:
        chdir(previous)


class StagnationAndPortfolioTests(unittest.TestCase):
    def test_parser_defaults_find_data_when_run_from_project_directory(self) -> None:
        with working_directory(BASE_DIR):
            args = weekly_pack._build_parser().parse_args([])

        self.assertTrue(Path(args.data_dir).exists())
        self.assertTrue(list(Path(args.data_dir).glob("function_*.csv")))

    def test_function_1_uses_low_signal_overrides_by_default(self) -> None:
        cfg = RebuildConfig()
        local_cfg = weekly_pack._apply_function_overrides(cfg, "function_1")

        self.assertFalse(local_cfg.use_output_power_transform)
        self.assertEqual(local_cfg.portfolio_primary, "global_explore")
        self.assertEqual(local_cfg.portfolio_when_stagnant, "global_explore")

    def test_stagnation_flag_uses_robust_y_scale(self) -> None:
        cfg = RebuildConfig(
            tr_stagnation_window=3,
            tr_stagnation_rel_tol=0.05,
            tr_stagnation_abs_floor=1e-18,
        )

        y_small = torch.tensor(
            [[0.0], [1e-12], [2e-12], [4e-12], [4e-12], [7e-12]],
            dtype=torch.double,
        )
        y_large = torch.tensor(
            [[0.0], [1000.0], [2000.0], [3000.0], [4000.0], [4000.0], [4000.000001]],
            dtype=torch.double,
        )

        self.assertFalse(weekly_pack._stagnation_flag(y_small, cfg))
        self.assertTrue(weekly_pack._stagnation_flag(y_large, cfg))

    def test_stagnation_hybrid_prefers_non_tr_global_explore_anchor(self) -> None:
        cfg = RebuildConfig(
            portfolio_when_stagnant="stagnation_hybrid",
            stagnation_hybrid_ucb_rel_tol=0.05,
        )

        result, named = select_portfolio(
            cfg=cfg,
            n_points=30,
            mu_t=np.array([4.0, 1.0, 0.5]),
            sigma_t=np.array([0.1, 0.8, 0.2]),
            ei=np.array([0.4, 0.2, 0.1]),
            pi=np.array([0.4, 0.2, 0.1]),
            ucb=np.array([4.2, 5.0, 0.9]),
            ts=np.array([4.3, 1.2, 0.4]),
            exploration_score=np.array([9.0, 8.0, 1.0]),
            stagnation_flag=True,
            candidate_origin=np.array(["tr", "global", "elite"], dtype=object),
            mu_t_tiebreak=np.array([4.0, 1.0, 0.5]),
            stagnation_scale=10.0,
        )

        self.assertEqual(named["global_explore"], 1)
        self.assertEqual(result.strategy_name, "global_explore")
        self.assertEqual(result.selected_index, 1)

    def test_stagnation_hybrid_prefers_trust_region_when_local_ucb_is_competitive(self) -> None:
        cfg = RebuildConfig(
            portfolio_when_stagnant="stagnation_hybrid",
            stagnation_hybrid_ucb_rel_tol=0.05,
        )

        result, named = select_portfolio(
            cfg=cfg,
            n_points=30,
            mu_t=np.array([4.0, 1.0, 0.5]),
            sigma_t=np.array([0.1, 0.8, 0.2]),
            ei=np.array([0.4, 0.2, 0.1]),
            pi=np.array([0.4, 0.2, 0.1]),
            ucb=np.array([4.8, 5.0, 0.9]),
            ts=np.array([4.9, 1.2, 0.4]),
            exploration_score=np.array([9.0, 8.0, 1.0]),
            stagnation_flag=True,
            candidate_origin=np.array(["tr", "global", "elite"], dtype=object),
            mu_t_tiebreak=np.array([4.0, 1.0, 0.5]),
            stagnation_scale=10.0,
        )

        self.assertEqual(named["trust_region_ts"], 0)
        self.assertEqual(result.strategy_name, "trust_region_ts")
        self.assertEqual(result.selected_index, 0)


if __name__ == "__main__":
    unittest.main()
