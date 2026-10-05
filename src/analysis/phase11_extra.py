"""Phase 11 additions from existing logs (no new runs, no new fits).

1. Time-to-threshold (validation MAE <= 62.24, A′'s mean best; post-hoc threshold):
   wall-clock from run start (E000 record timestamp) to the completion of the first experiment that
   reaches it (record timestamp + its evaluation runtime), plus the experiment-count version.
2. Transfer 2×2 per matched seed pair {A′ features, B features} × {XGBoost, TabPFN}, from transfer.csv,
   with the gap decomposed in both orderings.

    uv run python -m src.analysis.phase11_extra
"""

from __future__ import annotations

import pandas as pd

from src.analysis.runs import SEEDS, experiments
from src.config import ANALYSIS_DIR, ROOT

OUT = ANALYSIS_DIR
THRESHOLD = 62.24


def time_to_threshold(threshold: float = THRESHOLD) -> pd.DataFrame:
    exp = experiments()
    rows = []
    for (arm, seed), g in exp.groupby(["arm", "seed"]):
        g = g.copy()
        g["ts"] = pd.to_datetime(g["timestamp"])
        start = g.loc[g["experiment_id"] == "E000", "ts"].iloc[0]
        g["done"] = g["ts"] + pd.to_timedelta(g["runtime_seconds"].fillna(0), unit="s")
        hit = g[(g["experiment_id"] != "E000") & (g["status"] == "ok") & (g["mae"] <= threshold)]
        end = g["done"].max()
        if len(hit):
            h = hit.iloc[0]
            rows.append({"threshold": threshold, "arm": arm, "seed": seed, "first_experiment": h["experiment_id"], "experiments": int(h["idx"]),
                         "hours_from_start": round((h["done"] - start).total_seconds() / 3600, 2),
                         "run_hours": round((end - start).total_seconds() / 3600, 2)})
        else:
            rows.append({"threshold": threshold, "arm": arm, "seed": seed, "first_experiment": "never", "experiments": None,
                         "hours_from_start": None, "run_hours": round((end - start).total_seconds() / 3600, 2)})
    return pd.DataFrame(rows)


def transfer_2x2() -> pd.DataFrame:
    t = pd.read_csv(OUT / "transfer.csv").set_index(["arm", "seed"])
    rows = []
    for s in SEEDS:
        a, b = t.loc[("a_prime", s)], t.loc[("b", s)]
        aX, aT, bX, bT = a["mae_xgb"], a["mae_tabpfn"], b["mae_xgb"], b["mae_tabpfn"]
        total = bT - aX
        rows.append({
            "pair": f"B/{s} vs A′/{s}", "A′feat_XGB": aX, "A′feat_TabPFN": aT, "Bfeat_XGB": bX, "Bfeat_TabPFN": bT,
            "total_gap": total,
            # ordering 1: model first (holding A′ features), then features (holding TabPFN)
            "o1_model": aT - aX, "o1_feature": bT - aT,
            # ordering 2: features first (holding XGBoost), then model (holding B features)
            "o2_feature": bX - aX, "o2_model": bT - bX,
            "interaction": (bT - bX) - (aT - aX),
        })
    df = pd.DataFrame(rows)
    mean = df.drop(columns="pair").mean().to_dict() | {"pair": "mean of 3 pairs"}
    df = pd.concat([df, pd.DataFrame([mean])], ignore_index=True)
    df["o1_model_share"] = df["o1_model"] / df["total_gap"]
    df["o2_model_share"] = df["o2_model"] / df["total_gap"]
    return df


def env2_thresholds() -> list[float]:
    """Env-2: the mean best validation MAE of A and of A′ (post hoc, like Env-1's 62.24)."""
    from src.analysis.runs import best_experiments

    best = best_experiments()
    return [round(best[best["arm"] == a]["mae"].mean(), 2) for a in ("a", "a_prime")]


def main() -> None:
    from src.config import ENVIRONMENT

    if ENVIRONMENT == "env2":
        ttt = pd.concat([time_to_threshold(t) for t in env2_thresholds()], ignore_index=True)
        ttt.to_csv(OUT / "time_to_threshold.csv", index=False)
        print(ttt.to_string(index=False))
        if (OUT / "transfer.csv").exists():
            tr = transfer_2x2(); tr.round(3).to_csv(OUT / "transfer_2x2.csv", index=False)
            with pd.option_context("display.width", 250):
                print(tr.round(3).to_string(index=False))
        return
    ttt = time_to_threshold().drop(columns="threshold")  # Env-1 file format as published
    ttt.to_csv(OUT / "time_to_threshold.csv", index=False)
    print(f"time to validation MAE <= {THRESHOLD}:\n", ttt.to_string(index=False))
    tr = transfer_2x2()
    tr.round(3).to_csv(OUT / "transfer_2x2.csv", index=False)
    with pd.option_context("display.width", 250):
        print("\ntransfer 2x2 + decomposition (MAE; negative = favours B / TabPFN):\n", tr.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
