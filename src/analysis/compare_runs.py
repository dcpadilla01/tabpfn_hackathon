"""Phase 11: trajectories, productivity, friction, paired bootstrap (Env-1).

    uv run python -m src.analysis.compare_runs
Writes experiments/analysis/{summary.csv, trajectories.png, bootstrap.json}.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis.runs import ARM_LABEL, ARMS, SEEDS, best_experiments, calls, experiments, predictions  # noqa: E402
from src.config import ANALYSIS_DIR, ROOT  # noqa: E402
from src.data.targets import KEYS, TARGET, load_targets  # noqa: E402

OUT = ANALYSIS_DIR
COLORS = {"b": "#2a6fdb", "a_prime": "#d1495b"}


def per_run_summary(exp: pd.DataFrame, cl: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (arm, seed), g in exp.groupby(["arm", "seed"]):
        agent = g[g["experiment_id"] != "E000"]
        c = cl[(cl["arm"] == arm) & (cl["seed"] == seed)]
        llm, tools = c[c["event"] == "llm_call"], c[c["event"] == "tool_call"]
        hours = agent["wall_clock_seconds"].sum() / 3600
        mtok = llm["uncached_equivalent_tokens"].sum() / 1e6
        valid = int((agent["status"] == "ok").sum())
        e000 = float(g.loc[g["experiment_id"] == "E000", "mae"].iloc[0])
        best = agent.loc[agent["status"] == "ok", "mae"].min()
        rows.append({
            "arm": arm, "seed": seed,
            "E000_mae": e000, "best_val_mae": best, "improvement_vs_E000": e000 - best,
            "valid_experiments": valid, "failed_experiments": len(agent) - valid,
            "hours": round(hours, 2), "uncached_mtokens": round(mtok, 3),
            "llm_cost_usd": round(llm["cost_usd"].fillna(0).sum(), 3),
            "valid_per_hour": round(valid / hours, 2), "valid_per_mtoken": round(valid / mtok, 2),
            "tool_calls": len(tools),
            "run_python_errors": int(((tools["tool"] == "run_python") & (tools["status"] == "error")).sum()),
            "rejected_code": int((tools["status"] == "rejected").sum()),
            "llm_calls": len(llm), "truncated_llm_calls": int((llm["finish_reason"] == "length").sum()),
            "generated_code_chars": int(agent["generated_code_size"].fillna(0).sum()),
            "median_eval_seconds": round(agent["runtime_seconds"].median(), 2),
            "tabpfn_credits": int(g["tabpfn_estimated_credits"].fillna(0).sum()),
        })
    return pd.DataFrame(rows)


def time_to_quality(exp: pd.DataFrame, cl: pd.DataFrame, thresholds=(63.0, 62.24, 61.5)) -> pd.DataFrame:
    """Experiments / tokens / hours each run needed to first reach a validation MAE threshold.
    62.24 = A′'s mean final best."""
    rows = []
    for (arm, seed), g in exp.groupby(["arm", "seed"]):
        g = g[g["experiment_id"] != "E000"].copy()
        llm = cl[(cl["arm"] == arm) & (cl["seed"] == seed) & (cl["event"] == "llm_call")]
        g["mtok"] = g["experiment_id"].map(llm.groupby("experiment_id")["uncached_equivalent_tokens"].sum()).fillna(0).cumsum() / 1e6
        g["hours"] = g["wall_clock_seconds"].fillna(0).cumsum() / 3600
        for th in thresholds:
            hit = g[(g["status"] == "ok") & (g["mae"] <= th)]
            first = hit.iloc[0] if len(hit) else None
            rows.append({"arm": arm, "seed": seed, "threshold": th,
                         "experiments": None if first is None else int(first["idx"]),
                         "mtokens": None if first is None else round(first["mtok"], 2),
                         "hours": None if first is None else round(first["hours"], 2)})
    return pd.DataFrame(rows)


def trajectories(exp: pd.DataFrame, cl: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for (arm, seed), g in exp.groupby(["arm", "seed"]):
        g = g[g["experiment_id"] != "E000"].copy()
        mae = g["mae"].where(g["status"] == "ok")
        g["best"] = mae.cummin().ffill()
        llm = cl[(cl["arm"] == arm) & (cl["seed"] == seed) & (cl["event"] == "llm_call")]
        tok_by_exp = llm.groupby("experiment_id")["uncached_equivalent_tokens"].sum()
        g["mtok"] = g["experiment_id"].map(tok_by_exp).fillna(0).cumsum() / 1e6
        g["hours"] = g["wall_clock_seconds"].fillna(0).cumsum() / 3600
        kw = dict(color=COLORS[arm], alpha=0.85, lw=1.6, drawstyle="steps-post",
                  label=f"{ARM_LABEL[arm]}" if seed == 0 else None)
        for ax, x in zip(axes, ("idx", "mtok", "hours")):
            ax.plot(g[x], g["best"], **kw)
    for ax, xl in zip(axes, ("experiment #", "uncached-equivalent tokens (M)", "wall-clock hours")):
        ax.set_xlabel(xl); ax.grid(alpha=0.25)
    axes[0].set_ylabel("best validation MAE so far")
    axes[0].legend(frameon=False)
    fig.suptitle("Best-so-far validation MAE per run (3 runs per arm; Env-1)")
    fig.tight_layout(); fig.savefig(OUT / "trajectories.png", dpi=130)


def cluster_bootstrap(diff: pd.Series, households: pd.Series, n: int = 5000, seed: int = 0) -> dict:
    """Mean of `diff` with a household-clustered bootstrap CI."""
    df = pd.DataFrame({"d": diff.values, "h": households.values})
    per_h = df.groupby("h")["d"].agg(["sum", "count"])
    s, c = per_h["sum"].to_numpy(), per_h["count"].to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(n, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return {"mean": float(diff.mean()), "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "p_b_not_better": float((boots >= 0).mean())}


def paired_bootstrap(best: pd.DataFrame) -> dict:
    """Per row: |err| of B's best minus |err| of A′'s best (negative = B better)."""
    t = load_targets()
    val = t[t["split"] == "validation"][KEYS + [TARGET]]
    errs = {}
    for _, r in best.iterrows():
        p = predictions(r["arm"], r["seed"], r["experiment_id"])
        m = val.merge(p, on=KEYS, how="left")
        assert m["prediction"].notna().all()
        errs[(r["arm"], r["seed"])] = (m["prediction"] - m[TARGET]).abs().to_numpy()
    hh = val["household_key"].reset_index(drop=True)
    out = {"pairs": {}}
    for sb in SEEDS:
        for sa in SEEDS:
            d = pd.Series(errs[("b", sb)] - errs[("a_prime", sa)])
            out["pairs"][f"b{sb}-a'{sa}"] = cluster_bootstrap(d, hh)
    arm_b = np.mean([errs[("b", s)] for s in SEEDS], axis=0)
    arm_a = np.mean([errs[("a_prime", s)] for s in SEEDS], axis=0)
    out["arm_mean_abs_error"] = cluster_bootstrap(pd.Series(arm_b - arm_a), hh)
    out["note"] = ("Household-clustered bootstrap over validation rows; captures row sampling, not LLM run-to-run "
                   "variance (3 runs per arm). Best-of-20 selected on validation, so absolute MAEs are optimistic.")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    exp, cl = experiments(), calls()
    summary = per_run_summary(exp, cl)
    summary.to_csv(OUT / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(summary.T.to_string())
        agg = summary.groupby("arm").agg(["mean", "std"]).round(3)
        print("\nper-arm mean (std):")
        print(agg[["best_val_mae", "valid_experiments", "hours", "uncached_mtokens", "valid_per_hour", "valid_per_mtoken",
                   "run_python_errors", "rejected_code", "generated_code_chars", "median_eval_seconds"]].T.to_string())
    trajectories(exp, cl)
    ttq = time_to_quality(exp, cl)
    ttq.to_csv(OUT / "time_to_quality.csv", index=False)
    print("\ntime to quality (first experiment reaching validation MAE <= threshold; blank = never):")
    print(ttq.pivot_table(index=["threshold", "arm"], columns="seed", values=["experiments", "hours"]).to_string())
    best = best_experiments(exp)
    boot = paired_bootstrap(best)
    (OUT / "bootstrap.json").write_text(json.dumps(boot, indent=2))
    print("\nbest per run:", best[["arm", "seed", "experiment_id", "mae", "n_features"]].to_dict("records"))
    print("\narm-level paired bootstrap (B − A′, MAE):", boot["arm_mean_abs_error"])
    for k, v in boot["pairs"].items():
        print(f"  {k}: {v['mean']:+.3f}  CI95 [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
