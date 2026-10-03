"""Convert data/raw/*.csv → data/interim/*.parquet.

Asserts each source file's md5 and row count against src/data/schema.py, lowercases
column names, checks the column set, and casts to the registry dtypes, so data/interim/
is provably derived from a known raw release. Idempotent; overwrites outputs.

    uv run python scripts/convert_raw.py
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import INTERIM_DIR, RAW_DIR  # noqa: E402
from src.data.schema import TABLES, assert_columns  # noqa: E402


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def convert(name: str) -> None:
    spec = TABLES[name]
    src = RAW_DIR / spec.file
    t0 = time.perf_counter()
    digest = md5(src)
    if digest != spec.md5:
        raise AssertionError(f"{name}: md5 {digest} != expected {spec.md5} — different raw release?")
    # Read every column as string first where the registry says category/string, so
    # mixed codes like display '0'..'9','A' are not coerced to numbers.
    header = src.open().readline().strip().split(",")
    as_text = {c: pa.string() for c in header if spec.dtypes.get(c.lower()) in ("category", "string")}
    df = pacsv.read_csv(src, convert_options=pacsv.ConvertOptions(column_types=as_text)).to_pandas()
    df.columns = [c.lower() for c in df.columns]
    assert_columns(name, df)
    if len(df) != spec.rows:
        raise AssertionError(f"{name}: {len(df)} rows != expected {spec.rows}")
    raw = df[list(spec.columns)]
    df = raw.astype(spec.dtypes)
    for c, dt in spec.dtypes.items():  # astype wraps integer overflow silently — refuse it
        if dt.startswith(("int", "float")) and not (raw[c].min() == df[c].min() and raw[c].max() == df[c].max()):
            raise AssertionError(f"{name}.{c}: cast to {dt} changed the value range")
    out = INTERIM_DIR / f"{name}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{name:18s} {len(df):>11,} rows  md5 ok  -> {out.name}  ({time.perf_counter() - t0:.1f}s)")


def main() -> None:
    for name in TABLES:
        convert(name)


if __name__ == "__main__":
    main()
