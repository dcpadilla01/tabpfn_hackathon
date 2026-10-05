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

## 2026-10-04 — Env-2 sandbox friction fix (from the Arm A smoke run)

The Arm A smoke run (seed 900, budget 3) had 11 rejected cells, 8 of 14 calls in one experiment, nearly all benign
library introspection. For **every arm in Env-2** (an Env-2 difference from Env-1):
- `__name__`, `__version__`, `__doc__` are readable; all other dunders stay blocked.
- `getattr(obj, "literal"[, default])` is allowed only with a string-literal name that passes the attribute rules;
  dynamic `getattr`, `setattr`, `delattr` stay blocked. New `view.table(name)` gives dynamic table access safely,
  and the rejection message points to it. The prompt gains one line documenting `view.table(name)`.
- `import time` is allowed.
Re-checked against the smoke run: 3 of 11 rejections would now pass; 7 were dynamic `getattr` over table names
(now served by `view.table`), 1 a file write (`np.save('/tmp/…')`), correctly still blocked. The 6 code errors in
that run were ordinary pandas mistakes, not harness-induced.

## 2026-10-05 05:51 CST — Env-2 frozen test: pre-declaration (no Env-2 test number computed yet)

- **Scope change, stated plainly.** Env-2 was declared as a validation-and-behaviour comparison (CLAUDE.md,
  "Environments and primary results"). A frozen test is being added **now, before any Env-2 test score has been
  computed**: `experiments/analysis_env2/` does not exist at the time of writing. That is why this is a
  pre-declaration, not a post-hoc choice.
- **Primary cross-arm regime: train-only** (fit on train, score test once), for A, A′ and B. Arm A can only be
  scored this way: its agents trained their own models on train labels.
- **Replication of Env-1:** A′ and B are also scored with the train + validation refit, Env-1's primary regime,
  reported as a replication, not as the Env-2 cross-arm result.
- **Arm A replay rule.** Each A run's final candidate (chosen on validation only) is re-run from its own code with
  test rows switched on. It must first reproduce the logged validation predictions; a run that cannot produce test
  predictions is reported as **not replayable**. No workarounds, no hand edits. Non-replayable runs are an
  expected, acceptable outcome (agents may hard-code snapshot-day thresholds).
- **Order of work:** effort split (three arms), time-to-quality in hours (three arms), validation bootstrap (three
  arms), transfer 2×2 (A′/B replication), frozen test A′/B, then Arm A frozen test as time permits.

## 2026-10-05 — Known rough edges left unchanged during Env-2 (same for every arm)

- `save_table("x")` accepts a name without `.parquet`, but `load_saved("x")` requires the suffix; the rejection
  message says so and agents recovered in one call. Not fixed mid-run, because each code cell reloads `agent_api`, so a
  change would have altered the environment partway through the runs.
- Tool output is truncated to 6,000 characters (head + tail), which can cut the middle of XGBoost's long C++ stack
  traces. Same rule in every arm and both environments.
- Env-2's wall-clock numbers come from nine concurrent runs on one 16 GB machine. A memory spike at 23:16
  (four heavy cells at once; one hit the 300 s timeout) lost no run. Cross-arm time comparisons within Env-2 are
  fair; comparisons with Env-1 (six concurrent runs) are not.
