# dunnhumby Complete Journey — data schema

Source: dunnhumby "The Complete Journey" source files and accompanying user guide (PDF, © dunnhumby, not redistributed here).
Release: eight CSVs identified by md5 in [`data/README.md`](../data/README.md); checked by `scripts/convert_raw.py`.
Machine-readable mirror: `src/data/schema.py` (this document and that file must agree; `scripts/check_schema.py` enforces it).

## Conventions

- All column names lowercased on load (the raw CSVs mix `UPPER` and `lower`).
- `day` is an integer index 1–711 (no calendar dates). `week_no` is 1–102.
- **Week mapping:** week 1 is days 1–5 (partial); from day 6 on, weeks are 7 days: `week_no == (day + 8) // 7`. Verified for every line item. (The naive `(day − 1) // 7 + 1` is wrong for 27% of rows.) Week 102 is days 706–711 (6 days).
- Temporal tables carry a `time_col` and are filtered `time_col <= as_of` by the accessor. Static tables have none.
- Grain = the column set that identifies one row. "Unique: no" means the source has duplicates on that key (kept as-is; see table notes).
- Money columns are `float64` (float32 cannot hold cents exactly; the conversion script refuses lossy casts).

## Entity relationships

| Left | Right | Join keys | Cardinality | Notes |
|---|---|---|---|---|
| transaction_data | hh_demographic | household_key | many → 0..1 | 801 / 2,500 households have demographics |
| transaction_data | product | product_id | many → 1 | 0 unmatched |
| transaction_data | causal_data | product_id, store_id, week_no | many → 0..2 | sparse; absence = no display, no mailer |
| transaction_data | campaign_table | household_key | many → many | 1,584 / 2,500 households received ≥1 campaign |
| transaction_data | coupon_redempt | household_key | many → many | 434 / 2,500 households ever redeemed |
| campaign_table | campaign_desc | campaign | many → 1 | 0 unmatched; `description` identical on both sides |
| coupon_redempt | campaign_desc | campaign | many → 1 | every redemption falls inside its campaign window |
| coupon_redempt | coupon | coupon_upc, campaign | many → many | 0 unmatched |
| coupon | campaign_desc | campaign | many → 1 | |
| coupon | product | product_id | many → 1 | 0 unmatched |

## As-of visibility rules

| Table | Visible to snapshot at `snapshot_day` when | Why |
|---|---|---|
| transaction_data | `day <= snapshot_day` | time_col |
| coupon_redempt | `day <= snapshot_day` | time_col |
| campaign_desc | `start_day <= snapshot_day` | a campaign exists once it starts; `end_day` may be in the future |
| campaign_table | its campaign's `start_day <= snapshot_day` | via campaign_desc |
| causal_data | `week_no <= week_of_day(snapshot_day)` | weekly grain |
| coupon | its campaign's `start_day <= snapshot_day` | static table, but campaign-bound |
| hh_demographic, product | always | static |

**Known residual look-ahead (documented, accepted):** `causal_data` is weekly, so a snapshot sees its whole current week's display/mailer rows. Snapshots step by exactly 4 weeks, so every snapshot sits on day 6 of its week (e.g. day 95 in week 14 = days 90–96): the look-ahead is **1 day** of exposure data for every snapshot. Display/mailer plans are published before the week starts, so a retailer would have had this information; it is not target information.

---

## Data tables

### transaction_data

- **File:** `transaction_data.csv`
- **Grain:** household_key × basket_id × product_id × day (line item). **Unique:** yes (0 duplicates).
- **Rows:** 2,595,732 **Households:** 2,500 **Baskets:** 276,484 **Day range:** 1–711 **Week range:** 1–102
- **time_col:** `day`
- Each basket belongs to exactly one household and one day.

| Column | Type | Description (from PDF) | Notes |
|---|---|---|---|
| household_key | int32 | Uniquely identifies each household | |
| basket_id | int64 | Uniquely identifies a purchase occasion | |
| day | int16 | Day when transaction occurred | snapshot axis |
| product_id | int32 | Uniquely identifies each product | → product |
| quantity | int32 | Number of the products purchased during the trip | 0 on 14,466 rows; up to 89,638 (fuel-like items) — unreliable as a volume measure |
| sales_value | float64 | Amount of dollars retailer receives from the sale | **target source**; ≥ 0; 0 on 18,850 rows |
| store_id | int32 | Identifies unique stores | |
| coupon_match_disc | float64 | Discount applied due to retailer's match of manufacturer coupon | ≤ 0 always |
| coupon_disc | float64 | Discount applied due to manufacturer coupon | ≤ 0 always |
| retail_disc | float64 | Discount applied due to retailer's loyalty card programme | ≤ 0 except 36 rows (> 0, source anomaly, kept) |
| trans_time | int16 | Time of day when transaction occurred | HHMM integer, 0–2359 |
| week_no | int8 | Week of the transaction. Ranges 1–102 | `(day + 8) // 7` |

**Decisions**
- `future_spend_4w = sum(sales_value)` over `(snapshot_day, snapshot_day + 28]`.
- Per the user guide, `sales_value` is what the retailer receives: it **already nets out** the loyalty-card discount (`retail_disc`) and the coupon match (`coupon_match_disc`), and it does **not** net out manufacturer coupons (`coupon_disc`, reimbursed to the retailer). Shelf price = `(sales_value − retail_disc − coupon_match_disc) / quantity`. We use `sales_value` unchanged: it is the retailer's revenue from the household, the quantity a retailer would forecast. Totals: Σ sales_value $8.06M; Σ retail_disc −$1.40M; Σ coupon_disc −$43k; Σ coupon_match_disc −$7.6k.

**Checks run** (`scripts/check_schema.py`)
- [x] columns match registry
- [x] week_no / day relation (`(day + 8) // 7`, all rows)
- [x] discount sign convention
- [x] household onboarding curve plotted → `docs/figures/households_active_per_week.png`

### hh_demographic

- **File:** `hh_demographic.csv`
- **Grain:** household_key. **Unique:** yes
- **Rows:** 801 **Coverage of transaction households:** 32.0% (801 / 2,500; every demographic household transacts)
- **time_col:** none (static; collection date unknown — treated as known at every snapshot)

The user guide's column table for this file is a copy-paste error (it repeats transaction-column names); the prose and the value sets below are authoritative. Values are **ordinal** except `classification_2`. Lexical order is wrong for several (`'Level10' < 'Level2'`).

| Column | Type | Values | Notes |
|---|---|---|---|
| household_key | int32 | | |
| classification_1 | category | `Age Group1` … `Age Group6` (6) | ordinal; age band |
| classification_2 | category | `X`, `Y`, `Z` (3) | not stated as ordinal |
| classification_3 | category | `Level1` … `Level12` (12) | ordinal; lexical sort is wrong |
| classification_4 | category | `1`, `2`, `3`, `4`, `5+` (5) | ordinal; household size |
| classification_5 | category | `Group1` … `Group6` (6) | ordinal |
| homeowner_desc | category | Homeowner, Probable Owner, Probable Renter, Renter, Unknown (5) | |
| kid_category_desc | category | `1`, `2`, `3+`, `None/Unknown` (4) | |

### campaign_table

- **File:** `campaign_table.csv`
- **Grain:** household_key × campaign. **Unique:** yes. **Rows:** 7,208 (1,584 households × 30 campaigns)
- **time_col:** none; visible via `campaign_desc.start_day`

| Column | Type | Description | Notes |
|---|---|---|---|
| household_key | int32 | Targeted household | |
| campaign | int16 | Campaign id 1–30 | → campaign_desc |
| description | category | Campaign type | TypeA, TypeB, TypeC; identical to campaign_desc.description |

---

## Lookup tables

### campaign_desc

- **File:** `campaign_desc.csv` · **Grain:** campaign · **Unique:** yes · **Rows:** 30 · **time_col:** `start_day`
- Campaigns start between day 224 and 659; end between 264 and 719 (some end after the data does). **No campaign is visible before day 224**, so campaign features are structurally empty for the first 5 train snapshots.

| Column | Type | Description | Notes |
|---|---|---|---|
| campaign | int16 | Campaign id | |
| description | category | Campaign type | TypeA/B/C |
| start_day | int16 | First day of the campaign | as-of column |
| end_day | int16 | Last day of the campaign | may exceed snapshot_day (and 711) |

### coupon_redempt

- **File:** `coupon_redempt.csv` · **Grain:** household_key × day × coupon_upc × campaign · **Unique:** yes · **Rows:** 2,318 · **time_col:** `day`
- 434 households; days 225–704; every redemption lies within its campaign's `[start_day, end_day]`.

| Column | Type | Description | Notes |
|---|---|---|---|
| household_key | int32 | Redeeming household | |
| day | int16 | Redemption day | |
| coupon_upc | int64 | Coupon identifier | → coupon |
| campaign | int16 | Campaign the coupon belonged to | |

### coupon

- **File:** `coupon.csv` · **Grain:** campaign × coupon_upc × product_id · **Unique:** no — 5,164 exact duplicate rows in the source, kept · **Rows:** 124,548 · **time_col:** none

| Column | Type | Description | Notes |
|---|---|---|---|
| campaign | int16 | Campaign id | |
| coupon_upc | int64 | Coupon identifier | 1,135 distinct |
| product_id | int32 | Product the coupon applies to | one coupon → many products |

### product

- **File:** `product.csv` · **Grain:** product_id · **Unique:** yes · **Rows:** 92,353 · **time_col:** none

| Column | Type | Description | Notes |
|---|---|---|---|
| product_id | int32 | Product identifier | |
| department | category | Top level of hierarchy | 44 |
| commodity_desc | category | Second level | 308 (incl. "NO COMMODITY DESCRIPTION") |
| sub_commodity_desc | category | Third level | 2,383 |
| manufacturer | int32 | Manufacturer code | 6,476 |
| brand | category | National / Private | 2 |
| curr_size_of_product | string | Package size, free text | 4,345 distinct; blank for some; not parsed |

### causal_data

- **File:** `causal_data.csv` · **Grain:** product_id × store_id × week_no · **Unique:** no — 15,245 rows where the same product/store/week appears on two display locations · **Rows:** 36,786,524 · **time_col:** `week_no` (weekly)
- Covers weeks 9–101, 115 stores, 68,377 products. Absence of a row = not on display and not in the mailer.

| Column | Type | Description | Notes |
|---|---|---|---|
| product_id | int32 | | |
| store_id | int32 | | |
| week_no | int8 | | |
| display | category | In-store display location | 0 not on display, 1 store front, 2 store rear, 3 front end cap, 4 mid-aisle end cap, 5 rear end cap, 6 side-aisle end cap, 7 in-aisle, 9 secondary location, A in-shelf |
| mailer | category | Weekly mailer placement | 0 not on ad, A interior page feature, C interior page line item, D front page feature, F back page feature, H wrap front feature, J wrap interior coupon, L wrap back feature, P interior page coupon, X free on interior page, Z free on front/back page or wrap |

**Note:** this is marketing *exposure* data; "causal" is dunnhumby's name, not a causal-inference claim. The agent sees it as display/mailer exposure.

---

## Prediction target (Phase 2)

`data/processed/targets.parquet`: `household_key, snapshot_day, future_spend_4w, split`. Built by `scripts/build_targets.py`; `--check` regenerates in memory and asserts a byte-identical content hash.

- **Snapshots:** every 28 days, anchored at day 683 (= 711 − 28, the last day with a complete horizon) and stepping back: 95, 123, …, 683 → 22 snapshots. Cadence chosen after the Phase 0 row-budget check (35k × 30 → 16.7 s per TabPFN API fit+predict).
- **Eligibility (row property, identical in every split):** future window complete, and the household's first *observed* transaction ≤ `snapshot_day − 84`.
- **Zero-spend windows kept** (target 0): dropping them would bias the target and make a median prediction meaningless for low-activity households.
- **Split:** by snapshot day, same boundaries for all households, 60/20/20 of snapshots (13/4/5). The last train label window ends on the first validation snapshot, so no validation label overlaps a training label window; same for validation → test.

| split | snapshots | days | rows | households | **zero share** | mean | median | p90 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| train | 13 | 95–431 | 26,437 | 2,497 | **20.7%** | 136.86 | 73.51 | 366.46 |
| validation | 4 | 459–543 | 9,989 | 2,498 | **19.3%** | 140.03 | 77.27 | 372.24 |
| test | 5 | 571–683 | 12,490 | 2,498 | **19.3%** | 146.39 | 83.75 | 387.58 |

Early train snapshots are small (onboarding): day 95 only includes households first seen by day 11.

---

## Gotchas log

| Date | Finding | Action |
|---|---|---|
| 2026-10-03 | transaction_data grain is line item, not basket | aggregate for spend |
| 2026-10-03 | `week_no ≠ (day − 1)//7 + 1` for 27% of rows; week 1 is days 1–5 | `week_of_day(day) = (day + 8) // 7`, verified on all rows |
| 2026-10-03 | Draft registry declared transaction duplicates; there are none | `grain_unique=True` |
| 2026-10-03 | This release has `classification_1–5` + `homeowner_desc` + `kid_category_desc` (no 6/7); PDF table mislabelled | registry fixed; values documented above |
| 2026-10-03 | `coupon` has 5,164 exact duplicate rows | kept; `grain_unique=False` |
| 2026-10-03 | `causal_data` has 15,245 product/store/week rows on two display locations | kept; `grain_unique=False` |
| 2026-10-03 | 36 rows with `retail_disc > 0` | kept; asserted count |
| 2026-10-03 | `quantity` up to 89,638 (fuel-like items) | use `sales_value` for spend |
| 2026-10-03 | float32 cannot hold cents exactly | money columns float64; lossy casts refused |
| 2026-10-03 | No campaign starts before day 224 | campaign features empty for early snapshots |
| 2026-10-03 | Only ~1,300 of 2,500 households shop in a given week after onboarding (~week 17) | zero-spend windows are ~20% of rows, not an edge case |
| 2026-10-03 | 34 households never transact after day 431 | they contribute all-zero targets in validation/test; kept by design |

## Open questions

- None blocking. `causal_data` mid-week visibility is documented above as an accepted residual.
