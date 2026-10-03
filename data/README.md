# Data

This project uses **dunnhumby — The Complete Journey**: two years of household-level
grocery transactions from 2,500 households, with demographics, product hierarchy,
direct-marketing campaigns, coupons and in-store display/mailer exposure.

> Data © dunnhumby. Used under dunnhumby's terms for the Source Files; not
> redistributed in this repository. Get it from dunnhumby:
> <https://www.dunnhumby.com/source-files/>. The user guide (PDF) is also distributed
> by dunnhumby and is not committed here.

## Setup

1. Download "The Complete Journey" CSV release from dunnhumby.
2. Place the eight CSV files in `data/raw/` (git-ignored):

| File | Rows (excl. header) | md5 |
|---|---:|---|
| `transaction_data.csv` | 2,595,732 | `e5d2df5bfb59b7d658ea496539be5ec0` |
| `hh_demographic.csv` | 801 | `8c99c72097c3fdb5612aa6fbbf38eacd` |
| `campaign_table.csv` | 7,208 | `0e52350f60d17d6a37d8f2020a7dff15` |
| `campaign_desc.csv` | 30 | `d65df30fa3f148ed5581486e1d4064da` |
| `coupon_redempt.csv` | 2,318 | `e5706268c103cdd92dadbe48106107d7` |
| `coupon.csv` | 124,548 | `8dbe67d1284fc66cba5c5f56cd305d17` |
| `product.csv` | 92,353 | `b53bbdd6cf49293c5a1927e56ad3157a` |
| `causal_data.csv` | 36,786,524 | `db6edee61cc369cc5c86db4cafe0f1fb` |

3. Convert and check:

```bash
uv run python scripts/convert_raw.py   # asserts md5 + row counts, casts dtypes → data/interim/*.parquet
uv run python scripts/check_schema.py  # grain, week mapping, joins, discount signs, onboarding curve
uv run python scripts/build_targets.py # → data/processed/targets.parquet (holds test labels; harness-only)
```

`convert_raw.py` refuses to run if any md5 or row count differs: a different
release (e.g. a Kaggle re-upload with renamed or extra columns) is caught here
rather than silently changing results. The hashes above are the release used for
every reported number.

## Layout

```text
data/raw/         original CSVs (git-ignored, never modified)
data/interim/     typed Parquet copies, one per table (git-ignored, derived)
data/processed/   targets.parquet: household_key, snapshot_day, future_spend_4w, split (git-ignored, derived)
```

Column definitions, grains, join keys and gotchas: [`docs/data_schema.md`](../docs/data_schema.md).
Machine-readable registry: [`src/data/schema.py`](../src/data/schema.py).
