from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent))

from config import RebuildConfig
from weekly_pack import run_weekly


def main() -> None:
    ap = argparse.ArgumentParser(description="Print one proposed portal input per function (CapstoneBO+)")
    ap.add_argument("--data_dir", default="CapstoneBO+/data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kernel_mode", choices=["baseline", "lin_matern32_add"], default="lin_matern32_add")
    ap.add_argument("--output_transform_mode", choices=["power", "copula"], default="power")
    ap.add_argument("--selection_mode", choices=["robust_consensus", "single_seed"], default="robust_consensus")
    ap.add_argument("--consensus_seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--consensus_min_successes", type=int, default=3)
    args = ap.parse_args()

    cfg = RebuildConfig(
        data_dir=args.data_dir,
        reports_dir="CapstoneBO+/reports",
        report_date=date.today().isoformat(),
        overwrite=False,
        seed=int(args.seed),
        kernel_mode=args.kernel_mode,
        output_transform_mode=args.output_transform_mode,
        selection_mode=args.selection_mode,
        consensus_seeds=tuple(int(x) for x in args.consensus_seeds),
        consensus_min_successes=int(args.consensus_min_successes),
    )

    _, _, portal_df, _ = run_weekly(cfg=cfg, write_outputs=False)

    print("Proposed portal inputs (6dp, hyphen-separated):\n")
    for _, r in portal_df.sort_values("function").iterrows():
        print(f"{r['function']}: {r['portal_string']}")


if __name__ == "__main__":
    main()
