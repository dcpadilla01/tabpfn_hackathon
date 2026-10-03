"""Phase 2: build and persist data/processed/targets.parquet (harness-only; holds test labels).

    uv run python scripts/build_targets.py          # build + print summary
    uv run python scripts/build_targets.py --check  # rebuild in memory, assert identical to disk
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import PROCESSED_DIR  # noqa: E402
from src.data.splits import SPLITS, assign_split  # noqa: E402
from src.data.targets import KEYS, TARGET, build_targets  # noqa: E402

OUT = PROCESSED_DIR / "targets.parquet"


def content_hash(df: pd.DataFrame) -> str:
    return hashlib.md5(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


def summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("split", observed=True)
    return pd.DataFrame({
        "snapshots": g["snapshot_day"].nunique(),
        "first_day": g["snapshot_day"].min(),
        "last_day": g["snapshot_day"].max(),
        "rows": g.size(),
        "households": g["household_key"].nunique(),
        "zero_share": g[TARGET].apply(lambda s: (s == 0).mean()).round(3),
        "mean": g[TARGET].mean().round(2),
        "median": g[TARGET].median().round(2),
        "p90": g[TARGET].quantile(0.9).round(2),
    }).reindex(list(SPLITS))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    df = assign_split(build_targets())
    assert not df.duplicated(KEYS).any(), "targets grain not unique"
    assert df["split"].notna().all()
    h = content_hash(df)
    if args.check:
        on_disk = pd.read_parquet(OUT)
        assert content_hash(on_disk) == h, "regenerated targets differ from data/processed/targets.parquet"
        print(f"deterministic: regenerated targets identical (hash {h})")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(summary(df).to_markdown())
    print(f"\nsnapshot days: {sorted(df['snapshot_day'].unique().tolist())}")
    print(f"{len(df):,} rows -> {OUT.name}  (hash {h})")


if __name__ == "__main__":
    main()
