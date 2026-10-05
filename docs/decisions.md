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

## 2026-10-03 — Researcher LLM

- **Decision (project owner, before any run):** the researcher uses **`z-ai/glm-5.3-flash` via OpenRouter**,
  temperature 0.7, max 16,000 tokens per call, no reasoning cap, through a provider-agnostic OpenAI-compatible
  client. `CLAUDE.md` originally named Claude Sonnet 5.5 on the Anthropic SDK; `PLAN.md` already named GLM 5.3
  Flash via OpenRouter. Both now agree with the README.
- **Reason:** cost (stated by the project owner; to be confirmed). Six 20-experiment runs cost ≈ $2.2 in total.
- **Determinism:** each request passes the run's seed (0, 1, 2); the provider does not guarantee deterministic
  outputs, and OpenRouter routed calls to different providers (logged per call). Hence ≥ 3 runs per arm.

## 2026-10-04 — Publication of derived data

- **Decision:** publish derived data for reproducibility under dunnhumby's research terms (owner confirmed the
  terms allow it): feature tables of the six selected runs (release `env1-artifacts`, 148 MB), label-free
  per-row predictions, and agent transcripts (which include printed excerpts). Raw files stay unpublished.

## 2026-10-04 — Env-2: what differs from Env-1 (branch `env-2`)

Env-2 reruns Arms B and A′ and adds Arm A, seeds 0–2, 20 experiments each, results in
`experiments/results_env2/`. Env-1 stays the primary result; the two are never mixed in one table.

**Differences (agent-visible unless noted):**
1. Run directory scrubbed from tool output (`<run>`), so the arm label is not visible (Env-1: visible in tracebacks).
2. `print` inside `fn` is shown at train snapshots and suppressed at validation snapshots; validation-snapshot
   errors are reduced to type + line (Env-1: prints inside `fn` never reached the agent; errors unfiltered).
   The prompt states this in one added sentence.
3. Shared prompt wording: `save_table` returns "a path to pass to the evaluation tool" (Env-1: "…to experiment()").
4. Arm A exists: tools `inspect`, `run_python`, `score`; sklearn, XGBoost, scipy importable; root E000 is the
   baseline table on the fixed XGBoost evaluator (a harness reference, identical to A′'s E000).
5. Subprocess environment, every arm: `LOKY_MAX_CPU_COUNT=2` added (affects only joblib users, i.e. Arm A).
6. Harness-only (not agent-visible): frozen-test replay mode; `EXPERIMENT_ENV` switch.

**Unchanged:** LLM (`z-ai/glm-5.3-flash`, temperature 0.7, max 16k tokens, no reasoning cap), budget (20),
15-call cap, 300 s per-cell timeout (already in Env-1 for every arm), `OMP_NUM_THREADS=1` (already in
Env-1), evaluator and backends, seeds 0–2, data, split, target.

**Launch plan (16 GB M1):** start B and A′ (6 runs, as in Env-1), check memory, then add Arm A's 3 runs.
