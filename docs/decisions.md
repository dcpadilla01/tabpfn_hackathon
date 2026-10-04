# Decisions log

Protocol decisions taken after work started, with the reason and what they do *not* change.
Earlier locked decisions are in `CLAUDE.md` ("Locked Decisions", "Environments and primary results").

## 2026-10-04 — Phase 12 primary regime

- **Decision:** the frozen-test primary result is the **train + validation** refit scored on test, as declared in
  `scripts/evaluate_test.py` (commit `0b18b36`) before any test number was computed. The train-only fit is
  reported beside it in a fitting-regime × arm 2×2 (README, Phase 12).
- **Considered and rejected:** promoting train-only (the exact validation-scored model; the plan's literal
  "evaluate once on test"). Rejected because it would be chosen after seeing both regimes, and it shows the
  larger B advantage (−1.39 vs −0.94).

## 2026-10-04 — Phase 12 replay exceptions (B/2, A′/2)

- **Decision:** keep both runs' replayed test numbers, flag them *approximate*, document each deviation, and add
  an exact-replays-only sensitivity table. No selected pipeline and no logged number changed.
- **B/2 evidence (Q: were the knots fit on train + validation covariates at research time? — Yes):** cell b/2
  step 347 (E018) computes `qs = np.quantile(lv, ...)` with `lv = np.log1p(...T[src]...)` over the whole table
  `T`; the cell printed `base shape (36426, 186)` = 26,437 train + 9,989 validation rows. Covariates only.
  The winning E019 table loads these columns from `e018_hinge_prune`; E019's own new knots (step 379) are fit on
  train rows only.
- **A′/2 evidence:** `index` equals row position (0…36,425 research; 0…48,915 replay) in a table sorted by
  (household_key, snapshot_day); Spearman with household_key = 1.0; increases with snapshot_day in 100% of
  households; derived from keys only. Test values are **not** generally outside the training range: 5 of
  12,490 test rows exceed the primary regime's fit-row range (the last household's final rows).
- **Sensitivity (exact replays, n = 2 per arm, primary regime, no CI):** B 64.16 vs A′ 65.04 on test,
  B − A′ = −0.88 (all runs: −0.94).
- **Regenerate:** `uv run python -m src.analysis.frozen_test` →
  `experiments/analysis/frozen_test_sensitivity_exact.csv`.

## 2026-10-04 — Phase 11 claim checks (from existing logs; no new runs or fits)

- **"Faster" — kept.** Time to validation MAE ≤ 62.24 (post-hoc threshold = A′'s mean best) agrees in both
  units: B runs 1–8 experiments / 0.02–0.87 h from run start; A′ one run at 17 / 2.17 h, two never.
  Timestamp-based hours match the cumulative-duration figures to within 0.06 h.
- **"≈¾ model, ¼ features" — withdrawn.** The transfer 2×2 (all four cells already existed in
  `transfer.csv`) gives a model share of 74% holding A′'s features but 238% holding B's (per pair: 86/104,
  104/146, 39/461). Criterion was agreement within ~10 percentage points; it fails, also without pair 2.
  Replaced by: the model effect favours TabPFN in all six cells; features are co-adapted to their backend
  (mean interaction −2.48 ≈ the total gap).
- **Regenerate:** `uv run python -m src.analysis.phase11_extra` → `time_to_threshold.csv`, `transfer_2x2.csv`.
