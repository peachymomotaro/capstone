from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THIS_WEEK_VALUES: dict[str, dict[str, object]] = {
    "function_1": {"x": [0.447536, 0.417979], "y0": 0.16480353058789365},
    "function_2": {"x": [0.712205, 0.079938], "y0": 0.5011538019755815},
    "function_3": {"x": [0.485830, 0.507406, 0.473061], "y0": -0.022127633983417855},
    "function_4": {"x": [0.418031, 0.410481, 0.406548, 0.416485], "y0": 0.7013049970924787},
    "function_5": {"x": [0.999999, 0.999999, 0.999999, 0.999999], "y0": 8662.405001248297},
    "function_6": {"x": [0.465348, 0.395642, 0.588016, 0.876307, 0.128649], "y0": -0.20676090918422793},
    "function_7": {"x": [0.206509, 0.061332, 0.409412, 0.307434, 0.302362, 0.655302], "y0": 3.1735200584156527},
    "function_8": {
        "x": [0.021711, 0.000000, 0.046566, 0.000000, 0.744925, 0.552628, 0.000000, 0.000000],
        "y0": 9.8015716336935,
    },
}

FLOAT_REL_TOL = 1e-12
FLOAT_ABS_TOL = 1e-15


def _has_matching_row(df: pd.DataFrame, candidate_row: dict[str, object], key_cols: list[str]) -> bool:
    if df.empty:
        return False

    mask = pd.Series(True, index=df.index)
    for col in key_cols:
        series = df[col]
        value = candidate_row[col]

        if col == "function":
            mask = mask & (series == value)
            continue

        series_num = pd.to_numeric(series, errors="coerce")
        value_num = float(value)
        mask = mask & pd.Series(
            np.isclose(
                series_num.to_numpy(dtype=float),
                value_num,
                rtol=FLOAT_REL_TOL,
                atol=FLOAT_ABS_TOL,
                equal_nan=False,
            ),
            index=df.index,
        )

    return bool(mask.any())


def append_old_bo_data(
    old_data_dir: Path | str,
    new_data_dir: Path | str,
    pattern: str = "function_*.csv",
) -> dict[str, int]:
    """Append rows from old BO CSVs into CapstoneBO CSVs.

    Rows are de-duplicated using all columns except ``index`` so repeated points
    are not added twice even if row numbering differs.

    Returns a mapping of filename -> number of rows added.
    """
    old_data_dir = Path(old_data_dir).expanduser().resolve()
    new_data_dir = Path(new_data_dir).expanduser().resolve()

    if not old_data_dir.exists():
        raise FileNotFoundError(f"Old data directory not found: {old_data_dir}")
    if not new_data_dir.exists():
        raise FileNotFoundError(f"New data directory not found: {new_data_dir}")

    added_by_file: dict[str, int] = {}

    for old_csv in sorted(old_data_dir.glob(pattern)):
        new_csv = new_data_dir / old_csv.name

        if not new_csv.exists():
            raise FileNotFoundError(
                f"Target CSV missing in new folder: {new_csv} (expected to match {old_csv.name})"
            )

        old_df = pd.read_csv(old_csv)
        new_df = pd.read_csv(new_csv)

        if list(old_df.columns) != list(new_df.columns):
            raise ValueError(
                f"Column mismatch for {old_csv.name}:\n"
                f"old={list(old_df.columns)}\nnew={list(new_df.columns)}"
            )

        key_cols = [c for c in new_df.columns if c != "index"]
        existing_keys = set(map(tuple, new_df[key_cols].itertuples(index=False, name=None)))

        rows_to_add = old_df[
            ~old_df[key_cols].apply(tuple, axis=1).isin(existing_keys)
        ].copy()

        if rows_to_add.empty:
            added_by_file[old_csv.name] = 0
            continue

        start_idx = (
            int(pd.to_numeric(new_df["index"], errors="coerce").dropna().max()) + 1
            if "index" in new_df.columns and not new_df.empty
            else 0
        )
        rows_to_add["index"] = range(start_idx, start_idx + len(rows_to_add))

        merged = pd.concat([new_df, rows_to_add], ignore_index=True)
        merged.to_csv(new_csv, index=False)

        added_by_file[old_csv.name] = int(len(rows_to_add))

    return added_by_file


def append_this_week_values(
    new_data_dir: Path | str,
    values: dict[str, dict[str, object]] | None = None,
) -> dict[str, int]:
    """Append one new row per function CSV using provided x values and y0."""
    new_data_dir = Path(new_data_dir).expanduser().resolve()
    if not new_data_dir.exists():
        raise FileNotFoundError(f"New data directory not found: {new_data_dir}")

    payload = values or THIS_WEEK_VALUES
    added_by_file: dict[str, int] = {}

    for function_id, point in payload.items():
        csv_path = new_data_dir / f"{function_id}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CSV for {function_id}: {csv_path}")

        df = pd.read_csv(csv_path)
        x_vals = list(point["x"])
        y0_val = float(point["y0"])

        x_cols = sorted([c for c in df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
        used_x_cols = [c for c in x_cols if df[c].notna().any()]

        if len(x_vals) != len(used_x_cols):
            raise ValueError(
                f"{function_id}: expected {len(used_x_cols)} x values, got {len(x_vals)}"
            )

        new_row = {col: pd.NA for col in df.columns}
        new_row["function"] = function_id
        new_row["y0"] = y0_val
        for i, col in enumerate(used_x_cols):
            new_row[col] = float(x_vals[i])

        # Duplicate check on function + populated x columns + y0.
        key_cols = ["function", *used_x_cols, "y0"]
        if _has_matching_row(df, new_row, key_cols):
            added_by_file[csv_path.name] = 0
            continue

        next_idx = (
            int(pd.to_numeric(df["index"], errors="coerce").dropna().max()) + 1
            if "index" in df.columns and not df.empty
            else 0
        )
        new_row["index"] = next_idx

        updated = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        updated.to_csv(csv_path, index=False)
        added_by_file[csv_path.name] = 1

    return added_by_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Append old BO data/function_*.csv into CapstoneBO data with dedupe."
    )
    parser.add_argument(
        "--old-data-dir",
        default="../Legacy Files/BOTorch/data",
        help="Path to old BO data folder (default: ../Legacy Files/BOTorch/data)",
    )
    parser.add_argument(
        "--new-data-dir",
        default="./data",
        help="Path to current CapstoneBO data folder (default: ./data)",
    )
    parser.add_argument(
        "--pattern",
        default="function_*.csv",
        help="Glob pattern to match CSV files (default: function_*.csv)",
    )
    parser.add_argument(
        "--append-this-week",
        action="store_true",
        help="Append the hardcoded this-week points in THIS_WEEK_VALUES.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.append_this_week:
        added = append_this_week_values(new_data_dir=args.new_data_dir)
    else:
        added = append_old_bo_data(
            old_data_dir=args.old_data_dir,
            new_data_dir=args.new_data_dir,
            pattern=args.pattern,
        )

    total = sum(added.values())
    print("Rows added by file:")
    for fname, count in added.items():
        print(f"  {fname}: {count}")
    print(f"Total rows added: {total}")


if __name__ == "__main__":
    main()
