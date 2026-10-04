# Project

**Working thesis:** What happens to autonomous data science when the cost of testing a predictive hypothesis becomes dramatically lower?

We test whether an autonomous ML researcher using TabPFN-3.5 can explore more useful data hypotheses under a fixed research budget than the same researcher using a conventional ML workflow — and we separate the contribution of the *model* from the contribution of the *harness*.

## Hackathon

Prior Labs TabPFN-3.5 Hackathon.

Official scope explicitly includes agents and creative applications of TabPFN-3.5. Submission requires a runnable project/repository; demo video is optional. Deadline: **Monday, October 6, 2026**. Work started Friday, October 2.

Official submission page:

https://platform.priorlabs.ai/hackathon-3.5

Before final submission:

- sign into Prior Labs
- join hackathon / accept terms
- submit runnable repository
- optionally attach demo video

---

# MVP vs. stretch (read this first)

Three working days. Build in this order and stop wherever the clock stops.

**MVP (must ship):**

```text
Phases 0–8
Arm B  (TabPFN primitive)        ≥3 runs
Arm A′ (XGBoost primitive)       ≥3 runs
Phase 11 comparison
Phase 12 frozen test
README
```

**Stretch, in order of value:**

```text
1. Arm A (free-form conventional researcher)  ← the only stretch item worth fighting for
2. Demo video
3. Tree / best-first search (Phase 13)
4. Second dataset (Phase 14)
```

Arm A is the only arm that measures "effort moves away from plumbing." Arm A′ is the only arm that isolates "what is TabPFN worth." If forced to choose, A′ ships and A does not.

---

# Core Experiment

Compare the same autonomous researcher under three environments. Same LLM, dataset, target, split, as-of data accessor, objective prompt, research budget, evaluation metric. The **only** difference between arms is the tool list.

## Arm A — Free-form Conventional Researcher (stretch)

Available:

- pandas
- sklearn
- XGBoost if desired
- ordinary Python/code execution
- the as-of data accessor (Phase 4a) — the only way to touch transactions
- `score(val_predictions)` — the only way to obtain a validation score (counts as one experiment)

Agent is responsible for preprocessing, feature engineering, encoding, model selection, configuration, debugging. Do not deliberately handicap it. Do not tell it to do HPO; let it solve the task naturally.

## Arm A′ — Harnessed Conventional Researcher (MVP)

Available:

- pandas
- the as-of data accessor
- fixed `experiment()` primitive with **XGBoost backend**

XGBoost configuration is fixed inside the evaluator: `objective="reg:absoluteerror"`, sensible defaults, optionally a small *fixed* tuning budget that the agent cannot change. Identical to Arm B except for the backend. One-line swap.

## Arm B — TabPFN Researcher (MVP)

Available:

- pandas
- the as-of data accessor
- fixed `experiment()` primitive with **TabPFN-3.5 backend**

No HPO.

## Readings

```text
A  vs A′   what the harness is worth
A′ vs B    what TabPFN is worth        ← the primary comparison
A  vs B    ecological comparison
```

## Prompt symmetry

One objective prompt, shared verbatim by all arms. It states the task, the metric, the budget, and the tools available. It does **not** list feature ideas, aggregations, windows, or joins for any arm. The research direction is the agent's to find.

---

# Compute

Development machine: MacBook Pro M1, 16 GB unified memory. No local GPU.

**Decision: TabPFN runs through the Prior Labs hosted API (`tabpfn-client`). Seeds are fixed everywhere.**

Rationale: MPS/CPU inference is only suitable for small datasets (local CPU caps around 5,000 rows; the package throttles MPS memory to 0.7 of the recommended max on Apple Silicon). The research loop needs 20 fits per run on ~35k rows; that is a GPU workload, and the API is the product being judged.

Layout:

| Component | Where |
|---|---|
| TabPFN fits (Arm B, transfer check) | Prior Labs API via `tabpfn-client` |
| XGBoost fits (Arm A′, Arm A) | local |
| LLM researcher loop | local, LLM via API |
| dunnhumby loading, accessor, targets | local (Parquet, avoid duplicate copies) |
| Local `tabpfn` package | Phase 0 smoke test only, few hundred rows |

Rules:

- Backend is a config switch: `tabpfn_backend: api | local`. Default `api`. A reviewer with a CUDA GPU can set `local` and reproduce.
- Fixed `random_state` for TabPFN, XGBoost, snapshot generation, split, and agent seed per run. Log the seed with every experiment.
- Record `tabpfn-client` version and the model version reported by the API in every experiment log (the client exposes model metadata after prediction).
- README states that reported numbers come from the API backend; small differences are expected across CPU/GPU/MPS even with a fixed seed.
- Before launching Arm B runs: check hackathon credits on the Prior Labs platform and estimate cost per fit × 20 × runs. The 50% token discount ended September 29.
- Phase 0 row-budget check is still mandatory, now measured as API wall-clock per fit at the planned train size (target: well under 2 min, or the "cheap hypothesis" loop is not cheap).
- Fallback if API cost/latency is a problem: rent a T4/L4/A10 for the weekend and run the evaluator there with `device="cuda"`.

---

# Dataset

Use downloaded **dunnhumby Complete Journey** data.

Raw information includes transactions, products, households/demographics, campaigns, coupons/redemptions, promotions.

Do not manually build a rich modeling dataset before the agent sees it. Transforming raw business history into useful predictive representations is part of the research task.

Known properties to respect:

- ~2,500 households, days indexed 1–~711 (no calendar dates; `snapshot_date` is a day index).
- Households appear gradually; the first months are onboarding and early history is thin.

---

# Prediction Task

Unit:

```text
household_id × snapshot_day
```

Target:

```text
future_spend_4w = total household spend in (snapshot_day, snapshot_day + 28]
```

Snapshot cadence: every 28 days by default. **Decide the cadence after the Phase 0 row-budget check** — with ~2,500 households and ~25 snapshots the train set is ~35k rows, which must fit TabPFN-3.5's context limits on the API and return in acceptable wall-clock. If it does not, use 56-day spacing or subsample train snapshots; record the decision.

Eligibility:

- complete 28-day future window
- household has **≥ 12 weeks of observed history** before `snapshot_day`

Primary metric: **MAE** on the point prediction.

Point prediction is loss-consistent in every arm: **median** of TabPFN's predictive distribution; XGBoost with an L1 objective. This is the evaluator's choice, not the agent's.

Secondary diagnostic: R².

---

# PHASE 0 — Repository + Environment

- [ ] Clean repository, `.gitignore`, environment config, Python ≥3.10.
- [ ] `pip install tabpfn-client`; authenticate; verify API access. Store the key in `.env` (never committed).
- [ ] `pip install tabpfn` for the local smoke test only.
- [ ] `TabPFNRegressor.fit/predict` smoke test with `output_type="median"` (or current equivalent — read the docs, do not guess).
- [ ] **Row-budget check (API):** fit TabPFN-3.5 via `tabpfn-client` on a synthetic table of the planned train size (~35k rows × 30 features), fixed seed, and record wall-clock and credits consumed. If it is rejected or takes >2 min per fit, revisit snapshot cadence. Log the model version the API reports.
- [ ] Confirm hackathon credits cover ≈ 20 fits × runs × arms that use TabPFN.
- [ ] Confirm dunnhumby files load. Record paths; do not commit raw data.

Done when `python scripts/smoke_test_tabpfn.py --backend api` and `--backend local` each perform one regression prediction, and the API row-budget timing is printed.

---

# PHASE 1 — Inspect Raw Dataset

Do not engineer predictive features.

- [ ] Every source file, table grain, primary/foreign keys.
- [ ] Day/time fields, household id, spend field, product hierarchy.
- [ ] Campaign / coupon / promotion relationships.
- [ ] Data ranges, missingness, **household onboarding curve** (households active per week).

Create three artifacts:

- `data/README.md` — download instructions, expected files, row counts, md5 per file, dunnhumby attribution. Raw CSVs in `data/raw/` are git-ignored. Schema PDF under `docs/source/` only if its terms allow redistribution; otherwise link.
- `docs/data_schema.md` — per-table grain, keys, column definitions transcribed from the PDF, join keys, gotchas.
- `src/data/schema.py` — machine-readable table registry (`file`, `grain`, `time_col`, joins) imported by the accessor and the evaluator's validation. Tables with `time_col=None` are static.

Known gotchas to document and check:

- `transaction_data` grain is line item (household × basket × product × day); spend needs aggregation.
- `day` 1–711, `week_no` 1–102. `causal_data` is weekly only → as-of filter `week_no <= week(snapshot_day)`.
- Target uses `sales_value`; document whether discounts are included and verify the sign of `retail_disc`, `coupon_disc`, `coupon_match_disc`.
- `hh_demographic` is anonymized (`classification_1`–`7`) and does not cover every household; check coverage.
- `campaign_table` has no date; a campaign is visible to a snapshot only via `campaign_desc.start_day <= snapshot_day`.
- `causal_data` = display/mailer exposure, not causal inference; name it accordingly for the agent.
- Lowercase all column names on load; assert each declared grain is unique.

Done when `python scripts/check_schema.py` passes grain-uniqueness and join-key checks and we know exactly how a household's history can be reconstructed as of any snapshot day.

---

# PHASE 2 — Canonical Target Table

```text
household_id, snapshot_day, future_spend_4w, split
```

Temporal split by snapshot day, same boundaries for all households:

```text
earliest ~60% → train
next ~20%     → validation
latest ~20%   → test
```

Never split rows randomly. Test labels hidden during research. Persist the table.

Done when regeneration is deterministic.

---

# PHASE 3 — Weak Root Baseline (E000)

Only household demographics + snapshot calendar info. No recent spend, recency, frequency, trends, category, promotion, campaign or rolling-window features — those are the search space.

Done when E000 produces a reproducible validation MAE.

---

# PHASE 4 — Immutable Evaluation Harness

## 4a. As-of data accessor (shared by all arms)

Leakage is enforced by construction, not checked afterwards. The only way any arm touches transactions is:

```python
history(household_id, as_of_day) -> transactions with day <= as_of_day
# or vectorised:
build_features(fn)  # calls fn(history_up_to_snapshot, snapshot_day) per (household, snapshot)
```

Static tables (demographics, products) are readable directly. Campaign/coupon tables are exposed through the same as-of filter on their date fields.

Any arm that bypasses the accessor is a protocol violation and the run is void.

## 4b. Evaluator

```python
evaluate(feature_table, backend="tabpfn"|"xgb") -> result
```

Evaluator owns: keys, target, train/validation split, schema validation, backend execution, loss-consistent point prediction, metric, logging. Caller cannot alter validation dates, test dates, target, metric, or backend config.

Returns:

```json
{"mae": 37.42, "r2": 0.31, "n_features": 8, "runtime_seconds": 5.8, "backend": "tabpfn"}
```

For Arm A, the equivalent primitive is `score(val_predictions)`: it checks the key set matches the validation split exactly, computes the metric, and logs. It never reveals validation labels.

## 4c. Logging

Every experiment:

```text
experiment_id, parent_id, arm, backend, hypothesis, transformation_description,
feature_columns, mae, r2, n_features, runtime_seconds, status, error,
tokens_in, tokens_out, tool_calls, generated_code_size, wall_clock_seconds
```

Done when `evaluate(build_baseline_features())` reproduces E000 for both backends.

---

# PHASE 5 — Validate Research Loop Manually

Run 2–3 hypotheses by hand through accessor → feature table → `evaluate()`:

```text
E001  Recent purchasing behavior predicts future spend.   + spend_last_4w
E002  Long-term value adds signal beyond recent behavior.  + spend_last_26w
E003  Promotion sensitivity.                               + share of spend on promoted items, last 12w
```

Run each under both backends. Done when three experiments run without touching the evaluator.

---

# PHASE 6 — Research Primitive

```python
experiment(candidate_feature_table) -> result
```

Validates the table, calls the fixed evaluator, returns metrics. Must NOT invent or select features, generate hypotheses, auto-join, choose windows, or tune. The intelligence stays in the agent.

---

# PHASE 7 — V0 Autonomous Researcher

Simplest sequential agent. No MCTS, no UCB, no multi-agent, **no agent framework**.

## Stack

- Hand-written loop on a provider-agnostic LLM call (~200 lines, `src/researcher/agent.py`): the `openai` Python SDK against an OpenAI-compatible endpoint (`base_url` + key in config), so swapping provider/model is a config change, not a code change.
- LLM: **GLM 5.3 Flash via OpenRouter** (`OPENROUTER_API_KEY` in `.env`), exact model string pinned in `config/default.yaml`, identical for all arms. Explicit `temperature`. The API has no sampling seed → the LLM is the one non-seeded component; this is why runs ≥3 per arm.
- No framework (Agent SDK, LangGraph, smolagents): their injected system prompts, tool descriptions and retry logic are uncontrolled protocol variables and defeat accessor-only access. Fallback only if the loop is not working by Saturday evening: smolagents `CodeAgent` with every template overridden and committed as a protocol artifact.

## Loop

```text
system:  objective prompt + schema.agent_view() + tool contract + budget rules   (frozen, cached)
turn:    experiment history (compact table: id, parent, hypothesis, mutation, mae, status) + remaining budget
step:    state hypothesis → choose parent → implement transformation via accessor → experiment()/score()
         → observe → reflect (concise rationale) → next
```

## Tools (arm = tool list; nothing else differs)

| Tool | A | A′ | B | Behaviour |
|---|---|---|---|---|
| `inspect(table, op, **kw)` | ✓ | ✓ | ✓ | head / describe / nunique / value_counts, through the accessor |
| `run_python(code)` | ✓ | ✓ | ✓ | subprocess; accessor pre-imported; returns stdout/stderr; sklearn/xgboost importable in A only |
| `experiment(table_path, hypothesis, parent, mutation)` | | ✓ | ✓ | fixed evaluator, backend per arm → metrics |
| `score(preds_path, hypothesis, parent, mutation)` | ✓ | | | validates key set == validation split, returns metric, never reveals labels |

Structured metadata is required on every `experiment`/`score` call:

```json
{"hypothesis": "...", "reasoning_summary": "...", "parent_experiment": "E003", "proposed_transformation": "..."}
```

The rationale fed back to the agent and stored in experiment records is concise only. Raw model reasoning text, if the provider returns it, goes to a separate audit log (`reasoning.jsonl`) that the agent never reads and that is never used as evidence for a claim.

## Bounds and accounting

- One experiment = one `experiment()` or `score()` call. Max **15 tool calls** between consecutive experiments; exceeding it logs a failed experiment and resets. Same cap in all arms so "an experiment" means the same thing.
- Per step, log: tool name, tokens in/out, cached tokens, wall-clock. Report **uncached-equivalent tokens** as the budget metric so prompt caching cannot favour an arm.
- Three logs per run in `experiments/results/<arm>/<seed>/`:
  - `calls.jsonl` — one line per LLM call or tool call; the single source of truth for cost (tokens incl. cached/reasoning, OpenRouter `generation_id`, cost if returned, tool status incl. rejections).
  - `transcript.jsonl` — full requests/responses and tool I/O, for audit and replay (reasoning text excluded).
  - `reasoning.jsonl` — raw reasoning text keyed by run/step/generation_id; git-ignored except one or two committed example runs.
  - `experiments.jsonl` token/tool-call fields are rollups computed from `calls.jsonl`, never counted separately.
- Each experiment also logs the TabPFN credit cost (`estimate_cost`, no quota consumed) so Arm B's API spend is counted.
- Phase 11 effort classification uses actions (tool calls, code, error/retry patterns), never reasoning text.
- History is the compact table, never generated code. Code is persisted to `experiments/results/<arm>/<seed>/E###/code.py` for audit.

## Accessor enforcement (tripwires, not a jail — disclose in README)

- `data/raw/` unreadable by the subprocess user; raw paths never appear in prompts or error text.
- `run_python` statically rejects code containing `read_csv`, `read_parquet`, `open(`, `glob`, `data/`, `os.listdir`; rejections are logged and count as a tool call.
- Post-run audit script greps all persisted code for the same patterns.

## Done when

`make run ARM=b SEED=0 BUDGET=3` completes three experiments end to end, every log field is populated, and `make run ARM=a_prime SEED=0 BUDGET=3` produces an identical trajectory shape with `backend=xgb`.

---

# PHASE 8 — Research Budget

V0 budget: **20 attempted experiments** per run, where an experiment is one call to `experiment()` or `score()`.

Also track tokens, wall-clock, valid vs failed experiments.

Compare arms under equal experiment-count, token, and wall-clock budgets. Experiment-count alone mechanically favours B; report all three.

---

# PHASE 9 — Build Arm A′ and Arm A

- Arm A′: config flag → backend `"xgb"`. Nothing else changes.
- Arm A (stretch): swap `experiment()` for `score()`; add code execution and sklearn/XGBoost; keep the accessor.

---

# PHASE 10 — Core Experiment Runs

```text
Arm B   ≥3 independent runs   (MVP)
Arm A′  ≥3 independent runs   (MVP)
Arm A   ≥3 independent runs   (stretch; 2 if that is what fits)
```

Never evaluate on test during research. Capture full trajectories.

---

# PHASE 11 — Primary Analysis

**Predictive result:** best validation MAE, improvement over E000, per run.

**Research productivity:** valid hypotheses per hour, per token.

**Engineering friction:** failed experiments, debugging attempts, tool calls, generated code size, runtime per experiment.

**Research behaviour:** classify each agent step as data hypothesis / feature construction / model engineering / preprocessing / debugging / evaluation; report effort share per arm.

**Representation transfer (cheap, informative):** take the best feature table from each B run and evaluate it with the XGBoost backend, and vice versa. Shows whether B found better *features* or TabPFN simply fits the same features better.

Report trajectories (score vs. experiment index, score vs. tokens) for every run, not only means. Three runs per arm is thin; say so.

Core question: *Does TabPFN move autonomous researcher effort away from ML plumbing and toward hypotheses about the data?*

---

# PHASE 12 — Frozen Test Evaluation

After all research is complete: select the final candidate per run using validation only; evaluate once on test. Report test MAE, test R², validation→test degradation. Never select on test.

---

# PHASE 13 — Search Extension (stretch, only if time)

Keep experiments as a tree (experiment, parent, hypothesis, transformation, score, children). Allow branching from earlier nodes. Optionally best-first or UCB. No full MCTS. Search is infrastructure, not the contribution.

---

# PHASE 14 — Second Dataset (stretch)

Same agent, same primitive, minimal prompt change, on another predictive dataset. Not required.

---

# PHASE 15 — Research Claims

Claims must match measured evidence.

Primary thesis:

> TabPFN reduces the cost of autonomous predictive experimentation by collapsing preprocessing, model selection and tuning into a reusable prediction primitive.

Stronger result, only if supported:

> Under an equivalent budget, the TabPFN researcher tests more valid data hypotheses and/or discovers stronger predictive representations than a conventional autonomous workflow.

**Equally valid outcomes — write the analysis so these read as findings, not misses:**

- A′ and B reach similar MAE, but B tests more hypotheses per token / per hour.
- B's features transfer to XGBoost with little loss (TabPFN helped the *search*, not the final model).
- The harness explains most of the A vs. B gap (A vs. A′ large, A′ vs. B small). This is still a result about where the cost of a hypothesis lives.

Do NOT claim: TabPFN eliminates data science; TabPFN always beats XGBoost; feature engineering is unnecessary; autonomous research is solved.

---

# PHASE 16 — Repository Packaging

```text
README.md
pyproject.toml / requirements.txt
data/README.md
src/
    data/        load.py  targets.py  splits.py  accessor.py
    evaluation/  evaluator.py  metrics.py  logger.py
    features/    baseline.py
    researcher/  agent.py  arms.py  prompts.py  history.py
    tools/       inspect.py  run_python.py  experiment.py  score.py  audit.py
    analysis/    compare_runs.py  transfer.py
experiments/results/
scripts/
    smoke_test_tabpfn.py  build_targets.py  run_baseline.py
    run_researcher.py --arm {a,a_prime,b} --seed N
    evaluate_test.py  compare.py
docs/data_schema.md
```

README: thesis, architecture (three arms, accessor, evaluator), dataset instructions, setup (API key via `.env`, backend switch), exact reproduction commands, protocol, results with trajectory plots, compute statement (numbers produced via the Prior Labs API, seeds fixed, model version logged; small cross-device differences expected), limitations (run count, single dataset, LLM variance), attribution (dunnhumby, Prior Labs/TabPFN).

---

# PHASE 17 — Demo (stretch)

2–4 minutes. Problem → hypothesis → live research loop (hypothesis → feature → TabPFN → score, several accumulating) → side-by-side counters per arm (hypotheses tested, valid, tokens, runtime, best held-out score) → takeaway.

---

# PHASE 18 — Submission QA

- [ ] Fresh clone works; dataset instructions work.
- [ ] No credentials committed; `.env.example` present; TabPFN API auth instructions included.
- [ ] Seeds and TabPFN model version appear in every experiment log.
- [ ] Baseline, researcher (each arm), evaluate_test and compare commands work.
- [ ] Cached results clearly identified as such.
- [ ] No fabricated numbers; figures regenerate from logs.
- [ ] Protocol described fairly, including what was cut.
- [ ] Attribution for dunnhumby and Prior Labs.
- [ ] Demo link if produced.
- [ ] Submit via Prior Labs platform before **October 6, 2026**.

---

# Locked Decisions (2026-10-03)

- **Environment:** `uv` + Python 3.12. Commit `uv.lock`. Pin `tabpfn`, `tabpfn-client`, `xgboost`, `openai` to exact versions; record their versions in every experiment log, not only the lockfile.
- **Eligibility:** household's *first observed transaction* day ≤ `snapshot_day − 84`. Applied identically in train, validation and test — a property of the row, not of the split.
- **Zero-spend windows:** kept (target = 0). Dropping them would bias the target and make the median prediction meaningless for low-activity households. Report the zero share per split in `docs/data_schema.md`.
- **Arm A′ XGBoost:** untuned, fixed config documented in the evaluator: `objective="reg:absoluteerror"`, `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, fixed seed. **No early stopping** (it would leak validation into fitting). Categoricals: evaluator casts to `category` dtype + `enable_categorical=True` — harness plumbing, never the agent's. README states both backends are untuned; the Phase 11 transfer check answers "XGBoost was handicapped."
- **Parquet interim:** `scripts/convert_raw.py` writes `data/interim/*.parquet`, casting to `schema.py` dtypes and asserting row counts and source md5, so interim data is provably derived from a known raw release.
- **LLM reasoning:** no reasoning cap (no effort or reasoning-token limit); `max_tokens: 16000` per call so reasoning is not truncated. Runs are slow (~12 min/experiment in smoke tests), so MVP runs execute in parallel overnight.
- **Access control:** no separate OS user. `run_python` subprocess gets a stripped env (`env={"PATH": ...}` only). Primary check: AST **import allowlist** (pandas, numpy, the accessor module; + sklearn/xgboost in Arm A only). Secondary tripwire: string-pattern denylist. Post-run audit. Disclosed as "static enforcement plus audit, not a sandbox."

# Environments and primary results (2026-10-04)

- **Env-1 (primary):** the overnight MVP runs, Arms B and A′, seeds 0–2. Phases 11 and 12 and the README are written against Env-1, whatever else happens.
- **Env-2 (bonus, optional):** the harness after the audit fixes (run path scrubbed; prints inside `fn` visible at train snapshots, suppressed at validation snapshots; validation-snapshot errors reduced to type + line). Arm A runs **only** in Env-2, and only together with B and A′ reruns in the same environment. Added as a second, fully controlled table if complete by Monday noon.
- **Never mix runs from different environments in one table.** Don't switch primary results late.

# README must include (from the 2026-10-04 audit)

- A **paragraph** (not a footnote) on the `fn` output gap: the route existed (validation snapshots see earlier validation label windows), nothing used it (prints never flushed; all 33 `fn` errors at train snapshots), closed by design with tests. See `docs/audit_label_access.md`.
- **Behaviour findings**, alongside the numpy modelling: four agents filtered `train_targets()` for validation days and got zero rows (most likely a wrong assumption, not intent; the protection held, and that's only known because it was checked); one agent deliberately mapped what works inside `fn`.
- One sentence: `gbm_pred` (b/2 E017) was fitted on train rows and predicted back onto them; not a leak, not a best table.
- **Effort relocation, quantified:** 380/923 executed cells (41%) fit the agent's own model on train labels in arms whose model was fixed; 607/923 (66%) touch labels at all.
- **Failure attribution:** 3 of B's 8 failures come from b/1 reusing the leaked run path (a harness bug, now fixed); valid experiments B 52/60, A′ 58/60; the remaining agent failures (5 vs 2) are too few to call an arm difference.

# Execution Order

```text
Fri   0 env + row-budget check · 1 inspect · 2 targets/split · 3 E000 · 4 accessor + evaluator (both backends)
Sat   5 manual loop · 6 primitive · 7 V0 agent · start Arm B runs · Arm A′ runs (overnight)
Sun   11 analysis + transfer · 12 frozen test · README · (Arm A if B/A′ are clean by noon)
Mon   QA · submit · demo only if everything else is done
```

Do not build agent infrastructure until target, split, accessor, evaluator, baseline and the manual loop are verified.

# Immediate Task

Start at Phase 0 and proceed through Phase 5. Run the row-budget check before fixing the snapshot cadence.