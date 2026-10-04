"""Protocol artifacts: the objective prompt (shared verbatim by every arm) and the tool
contract (generated from the arm's tool list; identical text for Arms A′ and B, which
share a tool list — the backend is deliberately not named).

The objective states the task, metric, budget and rules. It does not suggest features,
aggregations, windows or joins: the research direction is the agent's to find.
"""

from __future__ import annotations

from src.data.accessor import AGENT_NAMES
from src.data.schema import TABLES
from src.tools.run_python import ALLOWED_IMPORTS, MODELLING_IMPORTS

OBJECTIVE = """\
You are an autonomous data scientist running a research project under a fixed budget.

# Task
Predict `future_spend_4w` for each (household_key, snapshot_day): the household's total spend
(sum of transactions.sales_value) on days snapshot_day+1 … snapshot_day+28. It is 0 when the
household buys nothing in that window. Data come from a grocery retailer: ~2,500 households,
about two years of history. Days are integer indices (day 1 = first day of the data); there are
no calendar dates.

# Evaluation
- Rows are (household_key, snapshot_day) pairs. Snapshot days are every 28 days.
- Train snapshot days: {train_days}
- Validation snapshot days: {val_days}
- A later test period exists; it is held out by the harness and you never see it.
- Score: mean absolute error (MAE) of predictions on the validation rows. Lower is better.
- A household has a row at a snapshot if its first observed purchase is at least 84 days earlier.

# Budget
- You have {budget} experiments. One experiment = one call to `{eval_tool}`.
- Between two experiments you may make at most 15 other tool calls. If you exceed that, the
  current experiment is recorded as failed and counts against your budget. Every tool result
  ends with a counter of the calls used so far.
- An invalid `{eval_tool}` call (bad file, wrong keys, missing metadata) also uses one experiment.

# What to do
Each experiment should test one clearly stated hypothesis about what in a household's data
predicts its future spend. Build on what earlier experiments taught you: name the experiment
you start from as `parent` and describe what you changed as `mutation`. Experiment E000 is the
starting point: household demographics plus snapshot calendar information only. Its feature
table is available as `baseline_features()`.

# Rules
- All data access goes through `agent_api` (pre-imported in run_python). Inside
  `build_features(fn)` each snapshot sees only data up to that snapshot day. Outside it, direct
  views (`snapshot`, `history`, `inspect`) are capped at day {visible_day}.
- Write tables only with `save_table`. Code that reads or writes files any other way, or imports
  modules outside the allowed list, is rejected without running (the rejection still counts as a
  tool call).
- Work through tool calls. Keep any text you write short.

# Data tables (access as attributes of a view, e.g. view.transactions)
{tables}
"""

TABLE_VISIBILITY = {
    "transactions": "rows with day <= snapshot day",
    "demographics": "static; covers about a third of households",
    "products": "static",
    "campaigns": "campaigns with start_day <= snapshot day",
    "campaign_targets": "households targeted by campaigns already started",
    "coupon_redemptions": "rows with day <= snapshot day",
    "coupons": "coupons of campaigns already started",
    "display_mailer": "rows with week_no <= the snapshot's week (week_no = (day + 8) // 7)",
}


def tables_section() -> str:
    lines = []
    for agent_name, table in AGENT_NAMES.items():
        spec = TABLES[table]
        lines.append(f"- {agent_name} ({TABLE_VISIBILITY[agent_name]}): {spec.description}")
        lines.append(f"  columns: {', '.join(spec.columns)}")
    return "\n".join(lines)


AGENT_API_DOC = """\
`agent_api` (already imported; also `pd` and `np`):
- build_features(fn) -> DataFrame. Calls fn(view, snapshot_day) once per train/validation
  snapshot; fn must return a DataFrame indexed by household_key. Returns household_key,
  snapshot_day and your columns for exactly the train+validation rows. Each fn call runs in
  isolation: it cannot keep state between snapshots or write files.
  print() inside fn is shown for train snapshots only; at validation snapshots output is
  suppressed and errors report only their type and line.
- view.households: the households needing a row at that snapshot. view.day, view.week.
- baseline_features() -> the E000 feature table (merge on household_key, snapshot_day).
- train_targets() -> household_key, snapshot_day, future_spend_4w for TRAIN rows only.
- snapshot(as_of_day=None) -> a view at as_of_day (default/max {visible_day}).
- history(household_key, as_of_day=None) -> one household's transactions.
- snapshot_days() -> {{"train": [...], "validation": [...]}}.
- save_table(df, name) -> path to pass to experiment(); load_saved(path) -> a table you saved earlier.
- KEYS, TARGET constants.
- describe_tables() -> table descriptions.
Each run_python call is a fresh process: nothing persists between calls except files saved
with save_table. Use print() to see output. Allowed imports: {imports}."""

TOOL_DOCS = {
    "inspect": "inspect(table, op, column=None, as_of_day=None, n=5): quick look at a table "
               "(op: columns, shape, head, describe, nunique, value_counts). No code needed.",
    "run_python": "run_python(code): execute Python with agent_api. Returns stdout/stderr.",
    "experiment": "experiment(table_path, hypothesis, parent, mutation, reasoning_summary): evaluate a "
                  "feature table with a fixed regression model and a fixed evaluation that you cannot "
                  "change: it fits on the train rows and reports MAE and R² on the validation rows. "
                  "The table needs household_key, snapshot_day and at least one feature column, for "
                  "exactly the train+validation rows (build_features guarantees this). Columns may be "
                  "numeric, bool, string or category; missing values are allowed; at most 500 features.",
}


def tool_schemas(tools: list[str]) -> list[dict]:
    s = {
        "inspect": {
            "description": TOOL_DOCS["inspect"],
            "parameters": {"type": "object", "properties": {
                "table": {"type": "string", "enum": list(AGENT_NAMES)},
                "op": {"type": "string", "enum": ["columns", "shape", "head", "describe", "nunique", "value_counts"]},
                "column": {"type": "string"},
                "as_of_day": {"type": "integer"},
                "n": {"type": "integer"},
            }, "required": ["table", "op"]},
        },
        "run_python": {
            "description": TOOL_DOCS["run_python"],
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
        "experiment": {
            "description": TOOL_DOCS["experiment"],
            "parameters": {"type": "object", "properties": {
                "table_path": {"type": "string", "description": "path returned by save_table"},
                "hypothesis": {"type": "string", "description": "the claim about the data this experiment tests"},
                "parent": {"type": "string", "description": "id of the experiment you build on, e.g. E000"},
                "mutation": {"type": "string", "description": "what you changed relative to the parent"},
                "reasoning_summary": {"type": "string", "description": "one or two sentences: why you expect this to help"},
            }, "required": ["table_path", "hypothesis", "parent", "mutation", "reasoning_summary"]},
        },
    }
    return [{"type": "function", "function": {"name": t, **s[t]}} for t in tools]


def system_prompt(arm_cfg: dict, budget: int, days: dict, visible_day: int) -> str:
    eval_tool = "experiment" if "experiment" in arm_cfg["tools"] else "score"
    imports = sorted(ALLOWED_IMPORTS | (MODELLING_IMPORTS if arm_cfg["allow_modeling_imports"] else set()))
    objective = OBJECTIVE.format(
        train_days=", ".join(map(str, days["train"])), val_days=", ".join(map(str, days["validation"])),
        budget=budget, eval_tool=eval_tool, visible_day=visible_day, tables=tables_section(),
    )
    tools = "\n".join(f"- {TOOL_DOCS[t]}" for t in arm_cfg["tools"])
    api = AGENT_API_DOC.format(visible_day=visible_day, imports=", ".join(imports))
    return f"{objective}\n# Tools\n{tools}\n\n{api}\n"


def turn_message(history_table: str, saved_tables: list[str]) -> str:
    saved = ", ".join(saved_tables) if saved_tables else "none yet"
    return (f"{history_table}\n\nSaved tables in your workspace (load_saved): {saved}.\n"
            "Code from earlier experiments does not carry over; saved tables do.\n"
            "State your next hypothesis, implement it, and call the evaluation tool.")
