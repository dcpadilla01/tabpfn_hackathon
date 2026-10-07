# CLAUDE.md — reproducing this study

This repository is a **finished study** (Prior Labs TabPFN-3.5 Hackathon, October 2026): one LLM researcher,
three arms that differ only in their tools (B: fixed TabPFN; A′: fixed untuned XGBoost; A: builds its own
models), two environments (Env-1: B and A′; Env-2: all three after harness fixes). Work here is reproduction
and inspection of the records, not extending the study.

Where things are:

- `README.md` — the write-up, setup (Quickstart) and every reproduce command (Reproduce).
- `docs/decisions.md` — every protocol decision, dated, including what was pre-declared before results.
- `docs/audit_label_access.md` — the leakage audit and the holes found and closed.
- `docs/build_plan_2026-10.md` — the original phase-by-phase build plan: history, not instructions.
- `data/README.md` — getting the dunnhumby CSVs and their md5 checksums.
- `PLAN.md` — the author's original plan; keep it as written.

## Three levels of reproduction

Work cheapest first; each level has its own done-condition.

1. **Regenerate** every reported table and figure from the stored logs, no API calls: README Reproduce steps
   1–2, then 5a (`EXPERIMENT_ENV=env1`) and 5c (`EXPERIMENT_ENV=env2`). Done when `git status` shows no
   changed tracked file.
2. **Replay** each run's final candidate on the held-out test period: README step 5b (`make artifacts` first;
   spends TabPFN credits, about an hour per environment). Done when every run's status, reproduction kind and
   test MAE match the committed `experiments/analysis*/frozen_test/<arm>_<seed>.json` (the `replay` field holds
   timings and is expected to differ).
3. **New runs**: `make run ARM=<b|a_prime|a> SEED=<3+> BUDGET=<n>`. LLM outputs are not deterministic, so a new
   run reproduces the *distribution* of results (compare against the per-run spread), never the exact numbers.

## Rules that keep the records trustworthy

- **The reported runs are records.** `experiments/results/` (Env-1), `experiments/results_env2/` (Env-2) seeds
  0–2, and `experiments/analysis*/` are what the README cites. New runs use seed 3 and up; the runner refuses to
  overwrite an existing run directory.
- **One environment per table.** `EXPERIMENT_ENV=env1|env2` selects which records the analysis reads (default
  `env2`, the current harness). Any harness change (prompt, tools, sandbox rules, `agent_api`) defines a new
  environment: its own results directory, an id added to the whitelist in `src/config.py`, and a dated entry in
  `docs/decisions.md` written before its results exist.
- **Leakage control** lives in `src/researcher/agent_api.py`, `src/data/accessor.py` and
  `src/tools/run_python.py`; `make test` covers it. Run it after touching any of them.
- **Numbers come from files.** Quote results from `experiments/analysis*/` outputs or regenerate them. Env-1
  numbers were produced at commit `1fb5872` (the Env-1 harness; check it out to rerun Env-1 exactly), Env-2
  numbers at `db3810c`.

## Practicalities

- **Cost of a live run** (Env-2 harness): `BUDGET=3` ≈ $0.05 of LLM and 30–40 min; `BUDGET=20` ≈ $0.35–0.40 of
  LLM and 3–4 h (Arm A ≈ 6 h). Arm B also spends ≈ 10–14k TabPFN credits per experiment; A and A′ run XGBoost
  locally.
- **Memory.** On a 16 GB machine, six concurrent runs are fine; nine caused a swap spike. Run long jobs from a
  separate git worktree so branch switches cannot change the code under a running job.
- **Waiting on background jobs:** match with a bracketed pattern (`pgrep -f "[r]un_researcher.py"`) so the wait
  loop never matches its own command line.
- **Shell:** zsh does not word-split unquoted variables (use arrays or explicit paths); macOS `/bin/bash` is 3.2,
  so scripts stay POSIX `sh`.
- **Artifacts:** `make artifacts [ENV=env1|env2]` downloads the best feature tables (GitHub releases,
  sha256-checked); only the transfer and replay checks need them.
