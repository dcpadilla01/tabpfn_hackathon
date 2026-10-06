# What does a cheap predictive hypothesis do to an autonomous data scientist?

**Prior Labs TabPFN-3.5 Hackathon entry.** We give the same LLM researcher the same data, task, prompt and
budget, and change one thing — the model behind its `experiment()` tool: **TabPFN-3.5** (Arm B) or a fixed,
untuned **XGBoost** (Arm A′). Then we measure what it finds, what it costs, and where its effort goes.

> **Working thesis.** TabPFN reduces the cost of autonomous predictive experimentation by collapsing
> preprocessing, model selection and tuning into a reusable prediction primitive.

> **Pinned results.** Env-1 numbers were produced from the code and stored results at commit
> [`1fb5872`](https://github.com/dcpadilla01/tabpfn_hackathon/tree/1fb5872), the Env-1 harness. Reproducibility
> from a fresh public clone was checked end to end on 2026-10-04 (at `d7bcbbf`; `1fb5872` only updated the
> repository URL). Env-2 numbers were produced at
> [`db3810c`](https://github.com/dcpadilla01/tabpfn_hackathon/tree/db3810c). Later commits change documentation and the
> artifact download, plus one harness fix (2026-10-05: the as-of view class now enforces the research horizon itself;
> see *The side channel we found*), which no agent code in either environment used. On 2026-10-05, at `08d8e4a`, a fresh public clone re-ran every replay and recompute command for
> both environments (step 5b): all 15 frozen-test verdicts, both transfer tables and E000 matched the committed results.

## Quickstart

**Requirements:** macOS or Linux (the harness uses `os.fork`; Windows is not supported), `git`, `make`, and
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) (it installs Python 3.12). Two API keys:
**TabPFN** from the [Prior Labs platform](https://platform.priorlabs.ai) and **OpenRouter** from
[openrouter.ai/keys](https://openrouter.ai/keys). The **dunnhumby Complete Journey** CSVs — see
[`data/README.md`](data/README.md).

```bash
git clone https://github.com/dcpadilla01/tabpfn_hackathon.git && cd tabpfn_hackathon
uv sync                                     # environment from uv.lock
cp .env.example .env                        # then paste TABPFN_API_KEY and OPENROUTER_API_KEY
# copy the eight dunnhumby CSVs into data/raw/
make data check targets test                # build + verify data, 58 tests (≈ 2 min)
EXPERIMENT_ENV=env1 uv run python -m src.analysis.compare_runs  # regenerate Env-1 headline results (no API)
make run ARM=b SEED=3 BUDGET=3              # a short live research run with TabPFN (≈ 30–40 min)
```

| Cost of a live run | LLM (GLM 5.3 Flash) | TabPFN API credits | Wall-clock |
|---|---|---|---|
| `BUDGET=3` | ≈ $0.05 | ≈ 40k (B only) | ≈ 30–40 min |
| `BUDGET=20` (as reported) | ≈ $0.35–0.40 | ≈ 200–280k (B only) | ≈ 2.9–3.7 h |

Arm A′ (`ARM=a_prime`) runs XGBoost locally and uses no TabPFN credits. Full reproduction steps are under
[Reproduce](#reproduce).

## Results at a glance (Env-1: 3 runs per arm, 20 experiments each)

| | **B — TabPFN-3.5** | **A′ — XGBoost** |
|---|---:|---:|
| Best validation MAE (mean ± sd over runs) | **60.72 ± 0.04** | 62.24 ± 0.24 |
| Paired, household-clustered bootstrap, B − A′ | **−1.52** MAE, 95% CI [−2.19, −1.02] | |
| Experiments / hours to reach A′'s mean best (62.24)³, per run | **3, 8, 1 / 0.11, 0.87, 0.02 h** | 17 / 2.17 h (1 of 3 runs; 2 never) |
| **Frozen test MAE** (fit train+val), mean ± sd | **64.09 ± 0.12** | 65.03 ± 0.18 |
| Paired bootstrap on test, B − A′ | **−0.94** MAE, 95% CI [−1.28, −0.61] | |
| Valid experiments / 60 | 52 | 58 |
| Valid hypotheses per hour / per M tokens | 5.50 / 6.49 | **5.80 / 7.58** |
| Effort on model engineering + debugging (share of tool calls) | 58% | 51% |
| LLM cost per run | ≈ $0.37 | ≈ $0.35 |

What the evidence supports:

1. **TabPFN makes the researcher better, fast — and it holds on the frozen test.** Every B run beats every A′
   run on validation, and B reaches the level A′ ends at within a median of 3 experiments (minutes), a level
   two of three A′ runs never reach. On the held-out test period the gap is **0.94 MAE** (1.52 on validation)
   and stays clear: all 9 run pairings favour B. The validation→test change mixes period drift (E000, with no
   search, loses 5–8 MAE), the fitting-regime change and two approximate replays; we do not separate them. "Fast" holds in
   both units: B runs reach 62.24 after 1–8 experiments and 0.02–0.87 hours from run start; one A′ run
   reaches it after 17 experiments and 2.17 hours, two never do.
2. **TabPFN beat XGBoost on every feature table, but the features themselves were tuned to their backend.**
   Swapping backends, TabPFN scored lower on all six best tables. Agent B's features helped TabPFN slightly on
   average and hurt XGBoost in every run, so the two decomposition orders give very different answers (74% vs
   238% "model"). B's lead can't be split into a model part and a feature part. Each agent's representations
   fit the evaluator it was selected on (see Phase 11 details).
3. **TabPFN did *not* make individual hypotheses cheaper, and did *not* move effort away from plumbing.**
   Per experiment, both arms cost the same tokens and time; B tests slightly *fewer* valid hypotheses per
   hour and per token. Agents in *both* fixed-model arms rebuilt the plumbing they were spared: **41% of all
   code they ran fitted their own models** (numpy ridge, least squares, two hand-written gradient-boosting
   implementations) to pre-screen features against train labels.

**Env-2** (below) reruns A′ and B, which replicate their Env-1 means to within 0.03 MAE, and adds the free-form
researcher, Arm A.

The headline finding is therefore narrower and, we think, more interesting than the thesis: *a strong
reusable primitive raises the quality ceiling and the speed to a good answer, but an autonomous agent with a
score to minimise will re-create model engineering around any fixed primitive.*

![Best-so-far validation MAE per run](experiments/analysis/trajectories.png)

### Phase 11 details

**Time to threshold** — first experiment whose validation MAE ≤ 62.24³; hours are wall-clock from run start
(E000 record) to the end of that experiment's evaluation.

| Run | First experiment ≤ 62.24 | Experiments | Hours from start | Run length (h) |
|---|---|---:|---:|---:|
| B/0 | E003 | 3 | 0.11 | 2.92 |
| B/1 | E008 | 8 | 0.87 | 2.97 |
| B/2 | E001 | 1 | 0.02 | 3.65 |
| A′/0 | never | — | — | 3.34 |
| A′/1 | E017 | 17 | 2.17 | 3.05 |
| A′/2 | never | — | — | 3.70 |

³ The threshold (62.24 = A′'s mean best validation MAE) was chosen after the runs, from their results; it
is a descriptive reference point, not a pre-registered target.

**Representation transfer 2×2** — each run's best feature table on both backends (validation MAE; seeds
paired by index, which carries no meaning since LLM outputs are not deterministic). Total gap = B features on TabPFN − A′
features on XGBoost. Negative = favours B / TabPFN.

| Pair | A′ feat · XGB | A′ feat · TabPFN | B feat · XGB | B feat · TabPFN | Total gap | Ordering 1: model (A′ feat) / feature (TabPFN) | Ordering 2: feature (XGB) / model (B feat) |
|---|---:|---:|---:|---:|---:|---|---|
| 0 | 62.45 | 60.93 | 62.52 | 60.67 | −1.78 | −1.52 / −0.26 (model 86%) | +0.07 / −1.85 (model 104%) |
| 1 | 61.98 | 60.71 | 62.54 | 60.76 | −1.22 | −1.26 / +0.05 (model 104%) | +0.56 / −1.78 (model 146%) |
| 2 | 62.29 | 61.69 | 67.93 | 60.73 | −1.56 | −0.60 / −0.96 (model 39%) | +5.64 / −7.20 (model 461%) |
| **mean** | 62.24 | 61.11 | 64.33 | 60.72 | **−1.52** | **−1.13 / −0.39 (model 74%)** | **+2.09 / −3.61 (model 238%)** |

- The **model effect is negative in all six cells** (both orderings, every pair): TabPFN fits either arm's
  features better.
- The **feature effect changes sign with the backend**: on TabPFN, B's features are better on average
  (−0.39; mixed by pair); on XGBoost, A′'s are (+2.09; +0.32 without pair 2). The interaction (−2.48 on
  average) is as large as the total gap: each arm's tables are co-adapted to the backend they were selected
  on. An earlier draft reported "≈¾ model, ¼ features"; that holds only in ordering 1 and is withdrawn.

### Frozen test (Phase 12)

Each run's final candidate was chosen on validation only and evaluated once on the 5 test snapshots
(12,490 rows). Test features did not exist during research, so `scripts/evaluate_test.py` replays every
code cell that saved a table up to the final experiment, with a harness-only switch that makes
`build_features` also emit test-snapshot rows (still as-of per snapshot), and checks that the replayed
train+validation rows reproduce the logged feature table before scoring. The agents' `assert` statements
(research-time row counts) are stripped; they compute nothing.

| Run | Final | Val MAE | Test MAE (fit train+val) | Test MAE (fit train) | Reproduction |
|---|---|---:|---:|---:|---|
| B/0 | E018 | 60.67 | 64.14 | 63.74 | exact |
| B/1 | E015 | 60.76 | 64.18 | 64.18 | exact |
| B/2 | E019 | 60.73 | 63.95 | 63.78 | approximate¹ |
| A′/0 | E019 | 62.45 | 65.22 | 65.70 | exact |
| A′/1 | E020 | 61.98 | 64.86 | 65.12 | exact |
| A′/2 | E009 | 62.29 | 65.00 | 65.04 | approximate¹ |

¹ Same rows and columns; some values differ because the agent's code computes them over the whole table.
See *Replay exceptions* below.

**Fitting regime × arm** (means over 3 runs; Δ = test − validation; E000 = the demographics + calendar
root on the same backend and regime):

| Fitting regime | Arm | Validation | Test | Δ | E000 val → test | E000 Δ | Test B − A′ (95% CI) |
|---|---|---:|---:|---:|---:|---:|---|
| **train + validation (primary)** | B — TabPFN | 60.72 | **64.09** | +3.37 | 92.53 → 98.64 | +6.11 | **−0.94** [−1.28, −0.61] |
| **train + validation (primary)** | A′ — XGBoost | 62.24 | 65.03 | +2.79 | 92.45 → 97.74 | +5.30 | |
| train only | B — TabPFN | 60.72 | 63.90 | +3.18 | 92.53 → 100.18 | +7.65 | −1.39 [−1.95, −0.95] |
| train only | A′ — XGBoost | 62.24 | 65.29 | +3.05 | 92.45 → 99.68 | +7.23 | |

- **Primary regime.** Refitting the selected feature table on train + validation and scoring test was
  declared the primary Phase 12 result in `scripts/evaluate_test.py` before any test number was computed,
  and it stays primary. The train-only fit (the exact model that was scored on validation) is reported
  beside it; it shows the larger B advantage, which is one reason we do not promote it after the fact.
- **Both regimes agree on direction:** B beats A′ on test in all 9 run pairings in either regime
  (household-clustered bootstrap CIs all below zero).
- **The test period is harder, not the agents overfit.** E000 — which involves no search at all — loses 5–8
  MAE from validation to test; the researched candidates lose ~3 in both arms.
- **Adding validation rows helps XGBoost more than TabPFN** (65.29 → 65.03 vs 63.90 → 64.09), so B's edge is
  1.39 with the training data the agents worked with and 0.94 with one more quarter of data — consistent with
  TabPFN's advantage being largest in the small-data regime. One data point; we do not generalise it.
- **"Evaluate once."** Each final candidate was scored in both regimes, once each. To save per-row predictions
  for the bootstrap, the identical deterministic scoring was re-run; every re-run reproduced the same numbers.
  No choice (candidate, features, regime) was made on test.

#### Replay exceptions

**B/2 (E019, val 60.73).** 29 hinge features `hg_{spend_84, spend_28, fwd28_mean, wk_avg_84, spend_56}_j =
max(log1p(x) − q_j, 0)` differ between the logged and replayed tables (max relative difference 0.068). The knots
`q_j` are quantiles of the feature computed over every row of the table the agent was working on
(cell b/2 step 347, E018: `lv = np.log1p(...T[src]...)`; `qs = np.quantile(lv, [0.15, …, 0.9])`; that table
printed `base shape (36426, 186)` = 26,437 train + 9,989 validation rows). So at research time the knots were
already fit on train + validation covariates; in the replay the same code also sees test-period covariates,
which moves the knots slightly. Only feature values (spend aggregates) enter the quantiles — no labels. The
original numbers are kept and the run is flagged *approximate*.

**A′/2 (E009, val 62.29).** The agent submitted a column `index` left over from `reset_index()`. It is the row
position in a table sorted by (household_key, snapshot_day): 0…36,425 at research time, 0…48,915 in the
replay, so values shift once test rows are interleaved. It is a recoding of the keys (Spearman 1.0 with
household_key; increasing with snapshot_day within every household) and carries no label or future
information; snapshot order was already a feature (`snapshot_day_index`). Because rows are household-major,
test rows interleave with training rows: only 5 of 12,490 test rows fall outside the primary regime's training
range of `index`. The original numbers are kept and the run is flagged *approximate*.

#### Replay-fidelity sensitivity

The four exact replays only (B/0, B/1, A′/0, A′/1), primary regime (fit train + validation). **n = 2 per arm**;
no confidence interval is reported for this subset.

| Arm | n | Validation | Test | Δ |
|---|---:|---:|---:|---:|
| B — TabPFN | 2 | 60.72 | 64.16 | +3.44 |
| A′ — XGBoost | 2 | 62.21 | 65.04 | +2.83 |
| **B − A′** | | −1.49 | **−0.88** | |

The test gap on exact replays (−0.88) is within 0.06 of the all-runs primary result (−0.94).

---

## Env-2: adding the free-form researcher (Arm A)

Env-2 reruns A′ and B and adds **Arm A**: the same researcher with no fixed model. It has sklearn, XGBoost and
scipy, chooses and trains its own models, and submits validation predictions to `score()` (one call = one
experiment). Env-2 uses the post-audit harness: run path scrubbed, prints inside `fn` visible at train
snapshots only, narrower sandbox rejections. The full list is in `docs/decisions.md`. Seeds 0–2, 20 experiments,
nine runs in parallel on one machine. **Env-2 is reported separately and never mixed with Env-1.** Wall-clock times
are comparable across arms within Env-2, but not with Env-1 (nine concurrent runs against six).

| Env-2 (3 runs per arm) | **A — free-form** | A′ — fixed XGBoost | **B — TabPFN** |
|---|---:|---:|---:|
| Best validation MAE (mean ± sd) | 61.58 ± 0.07 | 62.27 ± 0.06 | **60.71 ± 0.08** |
| Hours per run | 6.33 | 3.67 | 3.76 |
| Valid experiments per hour / per M tokens | 2.73 / 5.38 | 4.87 / 8.23 | 4.53 / 7.93 |
| Cell timeouts (300 s) per run | 2.3 | 0.3 | 0.3 |
| First reaches A's mean best (61.58)⁴ | 2.7 h, 7.0 h, never | never (×3) | **E001, 1–5 min (×3)** |
| First reaches A′'s mean best (62.27)⁴ | 0.5 h, 0.3 h, 1.25 h | 2.42 h (1 run); never (×2) | **E001, 1–5 min (×3)** |

Validation, paired household-clustered bootstrap (arm level): **B − A = −0.87** [−1.64, −0.37];
B − A′ = −1.56 [−2.19, −1.09]; **A′ − A = +0.69** [+0.37, +1.01]. B and A′ replicate their Env-1 means to
within 0.03 MAE.

⁴ Post-hoc thresholds (each arm's mean best). B reaches both with its first experiment in every run.

**Reading the three arms.**
- **What the harness is worth (A vs A′):** negative here. The free-form researcher beats the fixed, untuned
  XGBoost by 0.7 MAE. *A primitive is only as good as its backend*: A′ shows what a mediocre fixed backend
  costs. That is also why B's lead over A′ (1.6) is larger than its lead over A (0.9).
- **What TabPFN is worth (A′ vs B):** 1.6 MAE on validation, as in Env-1.
- **The overall comparison (A vs B):** B beats the researcher that is free to build its own models, reaches A's
  final level with its first experiment, and spends about 60% of A's time per run. A's lower productivity
  includes its cell timeouts (2.3 per run against 0.3), so the gap is not purely agent behaviour.

**Effort — one classifier, both environments.** Each tool call is labelled once, by the action-based rules in
`src/analysis/effort.py`. Because one label per call puts a cell that builds features *and* fits a model under
"model engineering", we also report the share of executed cells that contain any model fit, which is the more
robust measure.

| Env | Arm | Model engineering | Debugging | Data exploration + feature construction | **Cells with any model fit** | Cells fitting on labels |
|---|---|---:|---:|---:|---:|---:|
| Env-1 | B | 27.1% | 31.3% | 29.2% | **44.9%** | 43.0% |
| Env-1 | A′ | 24.8% | 25.9% | 35.7% | **39.5%** | 39.3% |
| Env-2 | B | 20.8% | 26.3% | 36.9% | **31.6%** | 31.6% |
| Env-2 | A′ | 21.2% | 27.1% | 38.5% | **32.1%** | 31.6% |
| Env-2 | **A** | **45.2%** | 26.9% | 18.0% | **56.8%** | 54.2% |

- **Direction:** given the plumbing, the free-form researcher spends far more of its effort on modelling. That
  shows on both measures (57% of cells fit a model against about 32%; 45% against about 21% by primary label).
  A fixed primitive does move effort away from plumbing, while the harnessed arms still rebuild some of it.
- **Between environments, the harnessed arms' model-fitting share fell** (cells with any fit: 45% → 32% for B,
  40% → 32% for A′). This is measured; the cause is not established. Restored prints would predict fewer
  debugging retries, but debugging fell for B (31 → 26%) and rose for A′ (26 → 27%). Rejected cells also fell
  (16 → 6 and 14 → 6 per run) under the narrower sandbox rules, and the two changes are confounded.

**Transfer (A′/B replication).** TabPFN scores lower than XGBoost on 4 of 6 best tables. The two exceptions
(A′ seeds 0 and 1: 64.62 and 64.75 on TabPFN against 62.32 and 62.20 on XGBoost) both contain `index`, a
row-number column left by `reset_index()` that recodes `household_key`. A diagnostic re-fit without that one
column (`experiments/analysis_env2/diagnostic_index_column.json`) gives 60.92 and 60.78 on TabPFN. TabPFN is
sensitive to an ID-like leftover column in a way XGBoost is not. Logged numbers are unchanged.

**Frozen test (pre-declared 2026-10-05 05:51, before any Env-2 test score).** Cross-arm primary: train-only, the
only regime Arm A can be scored in (its agents fit on train labels). Replication: train + validation for A′ and B.

| Regime | Arm | Runs scored | Validation | Test | Δ | Test vs B (95% CI) |
|---|---|---:|---:|---:|---:|---|
| **train-only (primary)** | **B** | 3 | 60.71 | **63.79** | +3.08 | — |
| **train-only (primary)** | A | 2 | 61.58 | 64.65 | +3.06 | B − A = −0.86 [−1.45, −0.45] |
| **train-only (primary)** | A′ | 2 | 62.25 | 65.69 | +3.45 | B − A′ = −1.91 [−2.55, −1.41] |
| train + validation (replication) | B | 3 | 60.71 | 63.96 | +3.25 | — |
| train + validation (replication) | A′ | 2 | 62.25 | 65.05 | +2.80 | B − A′ = −1.09 [−1.44, −0.75] (Env-1: −0.94) |

Per run: B/0–2 exact reproductions. A/0 and A/2 exact. **A/1 not replayable**: it reproduced its validation
predictions exactly, but its code predicts validation rows only, so no test predictions were produced. A′/2 exact.
A′/1 approximate (`index`). **A′/0 not reproduced**: a cell that failed at research time succeeded in the replay,
and the tables diverged. Per the pre-declared rule there were no workarounds and no hand edits; the means above use
the runs that produced a test score. A′ − A on test = +1.05 [+0.71, +1.40].

## Design

```text
                 ┌──────────── identical in every arm ────────────┐
 LLM researcher ─┤ objective prompt · budget · as-of data API      ├─► experiment(table) ─► fixed evaluator
 (GLM 5.3 Flash) │ inspect · run_python (pandas/numpy only)        │        backend: TabPFN-3.5 (B)
                 └─────────────────────────────────────────────────┘                 XGBoost   (A′)
```

- **Task.** dunnhumby *The Complete Journey*: ~2,500 households, 711 days. Predict
  `future_spend_4w` = a household's spend in the 28 days after a snapshot day. 22 snapshots every 28 days;
  temporal split by snapshot (13 train / 4 validation / 5 test); zero-spend windows kept (~20% of rows).
  Metric: **MAE**; point prediction is loss-consistent in both arms (TabPFN median; XGBoost L1 objective).
- **Arms differ only in the backend.** A′ and B receive byte-identical prompts and tool descriptions; the
  backend is never named. XGBoost is fixed and untuned (`reg:absoluteerror`, 500 trees, lr 0.05, depth 6, no
  early stopping); TabPFN-3.5 is used with defaults through the Prior Labs API. Neither is tuned — the
  transfer check is what answers "XGBoost was handicapped".
- **Budget.** 20 experiments per run; at most 15 other tool calls between experiments (exceeding it fails
  the experiment); invalid `experiment()` calls also consume budget. Context resets every experiment: the
  agent sees a compact history table, never earlier code.
- **E000** (demographics + snapshot calendar only) is the common root: MAE 92.5 on both backends.
- **No agent framework.** A ~200-line hand-written loop (`src/researcher/agent.py`) on an OpenAI-compatible
  client pointed at OpenRouter, so no injected prompts or retry logic become protocol variables.

### Leakage control (by construction, then audited)

- **As-of accessor.** All data access goes through `agent_api`. `build_features(fn)` calls `fn(view, day)`
  per snapshot with every table filtered to what was knowable that day, each call in a **forked child**
  process so state cannot flow between snapshots. Direct views (`snapshot`, `history`, `inspect`) are capped
  at day 459, the first validation snapshot. `train_targets()` returns **train labels only**.
- **Static enforcement plus audit, not a sandbox.** `run_python` checks code before running it: an AST
  import allowlist (pandas, numpy, stdlib basics), blocked file I/O, reflection and private API access, and
  a string-pattern tripwire; the subprocess gets a stripped environment and scrubbed output. Every executed
  cell is re-audited after the run (`make audit`: clean).
- **Evaluator owns everything else**: keys, target, split, encoding, backend config, metric, logging.

### The side channel we found — and closed

At validation snapshot 543 the as-of view legitimately contains the label windows of validation snapshots
459/487/515. Only `fn`'s returned DataFrame is meant to leave the child process, but two other routes
existed in the MVP environment: anything `fn` **printed**, and **exception messages** raised inside `fn`.
Our post-run audit (`docs/audit_label_access.md`) found that **neither was used**: prints inside `fn` never
reached the agent at all (the child exited without flushing stdout — which also silently hid the agents'
own debug output, equally in both arms), and all 33 `fn` failures happened at a train snapshot, before any
validation snapshot ran. That is a closed route by accident and ordering, not by design. It is now closed
**by design, with tests**: inside `fn`, output is visible at train snapshots and suppressed at validation
snapshots, and validation-snapshot errors are reduced to type and line. Runs after this fix form a separate
environment (Env-2) and are never mixed with the results above.

**A second route, found after both environments had run.** `snapshot()` and `history()` refuse any day after the
research horizon (459), but the view class they wrap, `AsOf`, was also exported to agent code, and the class itself
had no cap: `AsOf(683)` would have returned data covering the validation and test label windows. The prompt never
mentioned it. An audit of all 2,650 code cells executed in both environments (smoke runs included) found **no
cell that calls or even names `AsOf`**, so no reported number is affected. It is now closed **by design, with
tests**: the cap lives in the view class (set only inside the agent's process; inside each `build_features`
snapshot, views are limited to that snapshot's own day), `AsOf` is no longer a public name, and every route we
could find (direct, imported, `type(view)(day)`, changing the limit) is refused. This change postdates both
environments' runs and the pinned commits below.

## Behaviour findings

- **Agents rebuilt model engineering around a fixed model.** 380 of 923 executed cells (41%) fit the
  agent's own model on train labels — ridge/least-squares sweeps, and in two B runs a gradient-boosting
  implementation written from scratch in numpy (allowlist: no sklearn) — almost always scored on a temporal
  pseudo-holdout inside train (fit ≤ snapshot 375/403, score 403/431). 607/923 (66%) touched labels at all.
  One such local proxy had a rank correlation of −0.10 with the real validation score.
- **Four agents reached for labels they were not given.** Four cells filtered `train_targets()` for
  validation snapshot days. They received zero rows (MAE `nan`). This most likely reflects a wrong
  assumption that the function covers every split rather than intent, but the behaviour is real: an agent
  with a budget and a score to minimise went looking for validation labels, and we only know the
  protection held because we checked.
- **One agent mapped its sandbox on purpose**, writing a probe that calls each API function from inside a
  snapshot and reports what is allowed. It found nothing beyond train labels.
- **Stacking without care.** One submitted feature (`gbm_pred`, B seed 2 E017) was fitted on train labels
  and predicted back onto the same train rows; not a validation leak, not a best table, and it scored worse
  than the run's best — the agent noticed, but its out-of-fold rewrite was never saved.
- **Failures.** B: 8 failed experiments, of which 3 came from one run reusing a run-directory path leaked in
  tracebacks (a harness bug, since fixed) and 5 were agent errors; A′: 2 agent errors. 5 vs 2 is too few to
  call an arm difference.

## Reproduce

Steps 1–3 and 5a need no research runs. Commands marked **API** spend TabPFN credits; **LLM** spends
OpenRouter credit.

```bash
# 1. setup
uv sync                                    # Python 3.12; exact pins in pyproject.toml / uv.lock
cp .env.example .env                       # TABPFN_API_KEY (Prior Labs), OPENROUTER_API_KEY
# 2. data — see data/README.md (download from dunnhumby; md5-checked)
make data check targets                    # typed parquet, schema checks, target table (hash-checked)
# 3. harness
make test                                  # 58 tests: leakage, sandbox, evaluator contract, agent loop
uv run python scripts/run_baseline.py --reproduce        # API: re-evaluates E000 on both backends vs the logs
uv run python scripts/smoke_test_tabpfn.py --backend api # API: smoke test + 35k-row timing
# 4. new research runs (≈ 3 h each at BUDGET=20, ≈ 6 h for Arm A). They use the current (Env-2) harness and
#    write to experiments/results_env2/. Seeds 0–2 hold the reported runs and are protected; use 3 and up.
#    To rerun the Env-1 harness exactly, check out commit 1fb5872.
make run ARM=b SEED=3 BUDGET=20            # API + LLM; also ARM=a_prime, ARM=a (LLM only)
make audit                                 # static re-check of every code cell the agents ran
# 5a. rebuild every Env-1 table and figure from the stored logs (no API calls; verified byte-for-byte on a
#     fresh clone). EXPERIMENT_ENV selects the environment; the default is env2 (the current harness).
export EXPERIMENT_ENV=env1
uv run python -m src.analysis.compare_runs # summary, trajectories, validation bootstrap, time-to-quality
uv run python -m src.analysis.effort       # action-based effort shares
uv run python -m src.analysis.phase11_extra # time-to-threshold, transfer 2×2 (from transfer.csv)
uv run python -m src.analysis.frozen_test  # Phase 12 2×2, sensitivity, test bootstrap
uv run python scripts/audit_labels.py      # label-access audit
# 5c. Env-2 (three arms) from its stored logs → experiments/analysis_env2/
export EXPERIMENT_ENV=env2
uv run python -m src.analysis.compare_runs && uv run python -m src.analysis.phase11_extra
uv run python -m src.analysis.effort_all   # one effort table over all 15 runs (both environments)
uv run python -m src.analysis.frozen_test_env2   # Env-2 frozen test + bootstrap per regime
# 5b. recompute the stored inputs of 5a and 5c by replay (optional; spends TabPFN credits)
make artifacts                             # both environments' best feature tables (GitHub releases, sha256-checked)
export EXPERIMENT_ENV=env1
uv run python -m src.analysis.transfer     # API: each best table on the other backend → transfer.csv
uv run python scripts/evaluate_test_e000.py # API: E000 on test, both regimes → frozen_test/e000.json
uv run python scripts/evaluate_test.py --accept-approx   # API: replay each final table (≈ 1 h), score test once
export EXPERIMENT_ENV=env2
uv run python -m src.analysis.transfer     # API: A′/B best tables on the other backend
uv run python scripts/evaluate_test.py --accept-approx --arm b         # API: replay + test, both regimes
uv run python scripts/evaluate_test.py --accept-approx --arm a_prime
uv run python scripts/evaluate_test_arm_a.py                           # Arm A: replay its own code; train-only
```

`scripts/evaluate_test.py` checks that each replayed table reproduces the logged one; without
`make artifacts` it still scores test but reports that check as *unchecked*. `evaluate_test_arm_a.py` checks
replayed validation predictions against the logged ones, which are in git, so it needs no download. `tabpfn.backend: local` in
`config/default.yaml` runs TabPFN on a local GPU instead of the API (the `tabpfn` package asks for a one-time
licence acceptance at <https://ux.priorlabs.ai>; the API key in `.env` is passed as `TABPFN_TOKEN`).

**Cached results.** Everything under `experiments/results/{b,a_prime}/{0,1,2}/` and `experiments/analysis/`
is the stored output of the reported runs (Env-1): logs, transcripts, the code the agents ran, and label-free
prediction files. Step 5a regenerates every table and figure from these files. The agents' feature tables
are not in git (1.7 GB); the six that the transfer and replay checks need are a release download
(`make artifacts`: one release per environment). Re-running the researcher (step 4) produces new trajectories: a seed is passed to the LLM, but the provider does not
guarantee determinism. The data are dunnhumby's, used for research under their terms; transcripts contain excerpts the
agents printed while exploring.

## Compute and versions

- TabPFN-3.5 via the Prior Labs API (`tabpfn-client` 0.6.1, model `v3.5_default`, checkpoint
  `tabpfn-v3.5-20260909`); 35k × 30 fit+predict in ~17 s. Every experiment log records the seed, the package
  versions and the checkpoint. Small differences across CPU/GPU/MPS are expected even with a fixed seed.
- XGBoost 3.4.1 locally (`hist`). LLM: `z-ai/glm-5.3-flash` via OpenRouter, temperature 0.7, no reasoning
  cap. Each request passes the run's seed (0, 1, 2), but the provider does not guarantee determinism, hence
  several runs per arm. Provider routing varied per call and
  is logged.
- Totals for the six MVP runs: ~16M tokens, ≈ $2.2 LLM cost, ≈ 0.9M TabPFN credits.

## Limitations

- **Three runs per arm, one dataset, one LLM.** The bootstrap CI reflects row sampling, not run-to-run LLM
  variance; with three runs per arm that variance is only roughly characterised (sd 0.04 vs 0.24).
- **Best-of-20 on validation is optimistic** (B's edge 1.52 on validation, 0.94 on test in the primary
  regime); the frozen test is the honest number. Two of six test numbers come from approximate replays (see the Phase 12 table).
- **Effort classification is rule-based on actions** (code patterns, retries), checked by hand on a sample;
  its "preprocessing" class is unreliable (≈3% of calls).
- **Arm A was run in Env-2 only,** with A′ and B rerun in the same environment; it is not comparable with Env-1
  rows. Env-2's frozen test scores 2 of 3 runs for A and A′ (one not replayable, one not reproduced).
- Not run: tree/best-first search over experiments; a second dataset.

## What we do not claim

TabPFN does not eliminate data science; it does not always beat XGBoost; feature engineering is not
unnecessary; autonomous research is not solved.

## Repository

```text
src/data/        schema registry, loaders, targets, splits, as-of accessor
src/evaluation/  evaluator, backends, metrics, experiment log
src/researcher/  agent loop, agent_api, prompts, arms, trace logging
src/tools/       run_python (static enforcement), inspect, experiment, audit
src/analysis/    compare_runs, transfer, effort
scripts/         convert_raw, check_schema, build_targets, run_baseline, run_manual,
                 run_researcher, evaluate_test, audit_labels, smoke_test_tabpfn
docs/            data_schema.md, audit_label_access.md, figures/
experiments/     results/<arm>/<seed>/ (logs, code, transcripts), analysis/
```

## Attribution

Data: **dunnhumby — The Complete Journey** (© dunnhumby; see `data/README.md`). The raw files are not
redistributed here. For reproducibility, data derived from them are published under dunnhumby's research
terms: per-household × snapshot feature tables for the six selected runs (release `env1-artifacts`),
label-free per-row predictions, and agent transcripts that include excerpts the agents printed while
exploring.
Model: **TabPFN-3.5** by **Prior Labs**. Built for the Prior Labs TabPFN-3.5 Hackathon.

**License:** the code in this repository is released under the [MIT License](LICENSE). The dunnhumby data and
data derived from it are not covered by that license; they remain subject to dunnhumby's terms.
