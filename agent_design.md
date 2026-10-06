## How the agent works

### Tools

| Tool | Arms | What it lets the agent do |
|---|---|---|
| `inspect` | all | Look at a table without code: columns, first rows, summary statistics, value counts. Only data up to day 459 (the start of the validation period). |
| `run_python` | all | Write and run Python in an isolated process and read what it prints. This is where features are built. Only approved libraries (pandas, numpy; Arm A also sklearn, XGBoost, scipy), no direct file access, 300 seconds per run. |
| `experiment` | B, A′ | Hand over a finished feature table. The fixed model trains on the training period and returns the validation MAE. **One call = one experiment.** |
| `score` | A | Hand over its own predictions for the validation period. Returns the MAE, never the answers. **One call = one experiment.** |

Inside `run_python`, all data access goes through one library, `agent_api`. The key function is **`build_features(fn)`**: the agent writes a function that turns *one household's history as of one date* into features, and the harness calls it once per snapshot date, showing it only data up to that date. 

The future is **hidden by construction**, not by trust. The agent can also read the **training-period answers** (`train_targets()`), save and reload its own tables (`save_table` / `load_saved`), and take capped direct looks at the data.

### Loop

A plain, hand-written loop (no agent framework):

1. **Start each experiment fresh.** The agent sees the instructions plus a *compact table of its past experiments*:
   ID, parent, hypothesis, what changed, score. It never sees its old code or conversation.
2. **Explore and build.** It calls `inspect` and `run_python` as often as it likes, up to **15 calls**. Every tool
   result ends with a counter (`[tool calls since last experiment: k/15]`).
3. **Submit.** It calls `experiment` or `score`, stating its hypothesis, which earlier experiment it builds on, what
   it changed and why.
4. **Repeat** until 20 experiments are used. Exceeding 15 calls, three replies without a tool call, or an invalid
   submission all count as a failed experiment.

The prompt is identical across arms except for the description of the scoring tool. 

LLM: GLM 5.3 Flash via OpenRouter, temperature 0.7.

---

## Deep Dive on an Experiment

### Arm B: hypothesis to score in about a minute

**Run B/0, experiment 1.** Starting point: E000, customer demographics and calendar only: MAE 92.53.

> **Hypothesis:** *"A household's recent purchasing behaviour (spend in trailing windows, trip frequency, recency,
> trend) predicts its next-4-week spend; …"*

| Step | What the agent did | Time |
|---|---|---:|
| 1 | `run_python`: print the table descriptions and snapshot dates; first attempt errors | 12 s |
| 2 | `run_python`: look at one snapshot's transactions and a household's history | 7 s |
| 3 | `run_python`: read the training answers: mean, median, share of zero-spend households | 7 s |
| 4 | `run_python`: write the feature function, build the table, save it | 21 s |
| 5 | `experiment`: TabPFN trains and scores | 20 s |

**Result: MAE 61.53**, a third better than the baseline, in **5 tool calls and about a minute**. The feature function
is ordinary pandas, the hypothesis translated directly into columns (abridged from `E001/code.py`):

```python
def make_feats(view, sd):                       # one snapshot date at a time; nothing after sd is visible
    tx = view.table("transactions")
    out = pd.DataFrame(index=view.households)
    for w in [28, 56, 84, 112, 364]:            # "spend in trailing windows"
        out[f"spend_{w}"] = tx[tx.day > sd - w].groupby("household_key").sales_value.sum()
    out["trips_28"] = ...                       # "trip frequency"
    out["days_since_last"] = ...                # "recency"
    out["trend"] = ...                          # "last 28 days vs the 28 before"
    return out

ft = build_features(make_feats)                 # the harness runs it per snapshot, past data only
path = save_table(ft, "e001_history")         # then the agent calls the experiment tool with this path,
                                                 # its hypothesis, and parent E000
```

### Arm A: the same idea, plus building the model

**Run A/0, experiment 13** (its best). By now the features were settled; the hypothesis was about the *model*:

> **Hypothesis (paraphrased):** averaging a diverse ensemble of XGBoost median-regression models trained on the raw
> target with one trained on the residual (target minus a persistence baseline) improves accuracy.

It took **14 tool calls and 20 minutes**. The first LLM call alone spent 4.5 minutes and 11,900 tokens planning.
Then came 13 `run_python` cells: training models (30–130 seconds each), checking them on a held-out slice of the
training period (fit on dates up to 403, check on 431), and fixing three errors. Finally `score`: **MAE 61.52**. B's first experiment had reached the same
level in about a minute.

That contrast is the whole study in miniature: **with a strong fixed model, a hypothesis about the data is one
feature function away from a score. Without one, every idea also needs a model built, tuned and debugged.**

---

## What the runs looked like

Each run is a chain of hypotheses, each building on an earlier one. A few real ones (validation MAE):

**B/0 (TabPFN):** recent spending, trips, recency (**61.53**) → *what* households buy: department mix, brands,
basket size (61.40) → marketing exposure and demographics (submission invalid) → … → **E013: drop 19 features
whose values drift out of the training range over time** (cumulative campaign counts, tenure, week indices):
**60.79**, the run's best. The best idea here was about the data, not the model.

**A/0 (free-form):** recent history + own XGBoost (**63.58**) → add marketing and demographics (67.20, worse) →
back to E001, switch the model's loss to absolute error (63.14) → … → **E013: diverse ensemble on raw and residual
targets: 61.52**. Most of the progress came from model changes.

**A′/1 (untuned XGBoost):** recent behaviour (**63.25**) → product mix (63.09) → marketing (63.14) → … → **E015: add
"stacked" features, predictions from the agent's own ridge model, fit leave-one-snapshot-out: 62.20**. With a weak
fixed model, the agent fed its own model's predictions in as features.

The full trajectories, code and transcripts for every run are in `experiments/results_env2/<arm>/<seed>/`
(see [Reading the results](#reading-the-results)).
