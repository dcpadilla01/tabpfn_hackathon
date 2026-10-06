# Label-access audit — MVP runs (Arm B, Arm A′; seeds 0–2)

Date: 2026-10-04. Scope: every executed `run_python` cell (923 cells), every `fn` passed to
`build_features` (190 bodies), and every submitted feature table in `experiments/results/{b,a_prime}/{0,1,2}`.
Re-run with `uv run python scripts/audit_labels.py`.

**Verdict: no agent observed a validation label by any route. The validation MAEs stand.**

## 1. What the API can return

| Route | Returns | Verified |
|---|---|---|
| `train_targets()` | labels for the 26,437 train rows (snapshots 95–431) only | row-for-row equal to the train split |
| `snapshot()` / `history()` / `inspect` | as-of data, capped at day 459 | max transaction day = 459; validation label windows start at day 460 |
| `build_features(fn)` | keys + agent features | no target column; each snapshot's `fn` runs in a forked child |
| `baseline_features()`, `snapshot_days()` | features / keys | no target column |
| `experiment()` | aggregate MAE and R² only | — |

## 2. What agents actually did

- **Local scoring used train labels only.** 607 of 923 cells touch labels. The dominant scheme is a
  temporal pseudo-holdout inside train: fit on snapshots ≤ 375/403, score on 403/431.
- **Four cells tried to select validation rows** from `train_targets()` (`snapshot_day == 459` / `>= 459`;
  b/0 s217, a′/1 s191, a′/2 s5, a′/2 s205). They received **zero rows** (`val rows 0`, MAE `nan`) or crashed first.
  Most likely a wrong assumption that the function covers all splits; blocked by construction either way.
- **Deliberate sandbox probing.** a′/0 E011 wrote a `probe` fn that calls `train_targets()`, `load_saved`,
  `history`, `snapshot(100)` from inside a snapshot and reports what works. Nothing beyond train labels was exposed.
- **Agents wrote their own models** in numpy inside the harnessed arms (ridge/least squares, and in b/1 and b/2 a
  hand-written gradient-boosting implementation) to pre-screen features. All on train labels.

## 3. Side channels out of `fn` (validation snapshots see earlier validation label windows)

At validation snapshot 543 the as-of view legitimately contains days ≤ 543, i.e. the label windows of
validation snapshots 459/487/515. Only the returned DataFrame is meant to leave the child.

| Channel | In the MVP runs | Since commit after this audit |
|---|---|---|
| Returned DataFrame | becomes features of that snapshot's own rows only (legitimate history) | unchanged |
| `print` inside `fn` | **never reached the agent** — the child exited via `os._exit` without flushing (all 8 checked markers absent) | shown for train snapshots, suppressed for validation snapshots |
| Exception messages from `fn` | reached the agent, but **all 33 `fn` failures occurred at a train snapshot** (snapshots run in order; bugs stop at day 95) | at validation snapshots reduced to `<ErrorType> at line N` |
| Files / globals | blocked (`save_table`/`load_saved` refuse in child; globals die with the child) | unchanged |

The print and error channels were closed in practice during the MVP runs, but by accident and by ordering,
not by design. They are closed by design for every later run (Arm A, reruns).

## 4. Stacked / model-prediction features

Five submitted experiments contain prediction-like features; **none is a run's best experiment**, so the
transfer check (best tables) does not depend on them.

| Run / exp | Feature | Built from | In-sample on train rows? |
|---|---|---|---|
| a′/2 E018 | `cyc_pred_*` | purchase-cycle heuristic on transactions | n/a (not label-fitted) |
| a′/0 E017 | `tp_pred` | activity × non-zero mean spend | n/a (not label-fitted) |
| a′/0 E012 | `emb_pred_*` | per-snapshot autoregressive ridge on the household's own past windows, inside `fn` | n/a (self-supervised, as-of) |
| b/2 E010 | `pred_2p` | per-snapshot two-part estimate from earlier anchors, inside `fn` | n/a (self-supervised, as-of) |
| b/2 E017 | `gbm_pred` | agent-written GBM + OLS fit on **all train labels**, predicted for all rows | **yes** — train rows in-sample, validation rows out-of-sample |

`gbm_pred` is not a validation leak but is careless: the evaluator's model sees an over-fitted feature on train
rows. The agent knew (`# simulate harness: … train rows in-sample`), attempted an out-of-fold version in E016
(table never saved → invalid experiment), and the in-sample version scored 61.84 vs. the run's best 60.73.

## 5. Other protocol notes found during the audit

- Tracebacks exposed `<harness>/experiments/results/<arm>/<seed>/…`, revealing the arm label (`b` / `a_prime`).
  The label does not name the backend. Scrubbed for later runs.
- Agent debug prints inside `fn` were silently lost in the MVP runs (same in both arms): a source of friction.

## Addendum (2026-10-05): uncapped view class

Found after both environments had run. `snapshot()`/`history()` capped direct access at day 459, but the class they
wrap, `AsOf`, was in the agent's public API without a cap: agent code could build `AsOf(683)` and see validation and
test label windows. Audit of every executed `run_python` cell in Env-1 and Env-2 (2,650 cells, smoke runs
included): **0 cells call `AsOf(…)`; 0 cells mention the name.** No reported result is affected.

Fix: the cap is enforced in `AsOf.__init__` via a class-level horizon that the runner locks to the research horizon
before agent code runs; each forked `build_features` child lowers it to its own snapshot day; harness-internal
alignment and the baseline table lift it for their own call only. `AsOf` was removed from the public API. The
horizon attribute is a dunder, so the static checker rejects any read or write of it. Tests cover direct,
imported, `type(view)(day)` and limit-changing routes (`tests/test_run_python.py`).

Verification after the fix: replays of Env-1 B/0 and Env-2 A/0 under the fixed harness reproduced their committed
frozen-test results exactly; no replayed cell changed outcome.
