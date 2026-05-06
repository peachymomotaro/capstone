from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch


@dataclass
class FunctionDataset:
    function_id: str
    csv_path: Path
    df_raw: pd.DataFrame
    x_cols: list[str]
    X: torch.Tensor
    y: torch.Tensor

    @property
    def d(self) -> int:
        return int(self.X.shape[-1])

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


def used_x_cols(df: pd.DataFrame) -> list[str]:
    x_cols_all = sorted([c for c in df.columns if c.startswith("x")], key=lambda s: int(s[1:]))
    cols = [c for c in x_cols_all if df[c].notna().any()]
    if not cols:
        raise ValueError("No populated x-columns found.")
    return cols


def load_function_dataset(csv_path: Path, dtype: torch.dtype = torch.double) -> FunctionDataset:
    df = pd.read_csv(csv_path)
    x_cols = used_x_cols(df)

    if "y0" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing y0")

    # Keep only complete rows for used x columns and y0.
    sub = df.dropna(subset=x_cols + ["y0"]).copy()

    X = torch.tensor(sub[x_cols].to_numpy(), dtype=dtype)
    y = torch.tensor(sub["y0"].to_numpy(), dtype=dtype).unsqueeze(-1)

    return FunctionDataset(
        function_id=csv_path.stem,
        csv_path=csv_path,
        df_raw=sub,
        x_cols=x_cols,
        X=X,
        y=y,
    )


def list_function_csvs(data_dir: Path) -> list[Path]:
    csvs = sorted(data_dir.glob("function_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No function CSVs found in {data_dir}")
    return csvs


def make_bounds(d: int, lower: float = 0.0, upper: float = 1.0, dtype: torch.dtype = torch.double) -> torch.Tensor:
    return torch.tensor([[lower] * d, [upper] * d], dtype=dtype)
