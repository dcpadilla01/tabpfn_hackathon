import agent_api, pandas as pd, numpy as np
print("SNAPDAYS", agent_api.snapshot_days())
tt = agent_api.train_targets()
print("targets shape", tt.shape)
print(tt.groupby('snapshot_day')['future_spend_4w'].agg(['count','mean','std','median']).round(1))

def L(nm):
    err=None
    for cand in (nm, nm+".parquet"):
        try:
            return agent_api.load_saved(cand)
        except Exception as e:
            err=e
    print("FAIL", nm, repr(err)[:120]); return None

names=["e009_ewma_longlags","e011_pruned","e013_full","e013_new_feats","e014_full_pool","e015_hh_identity","e016_dec_predictors","e017_hazard_ext","e017_hazard_only","e017_season_hazard","e017_dept_season","e018_cycle_projection","e008_main_plus_marketing","temporal_structure","basket_tenure","recency_agg","dept_mix_recency","weekly_history","marketing_exposure","product_habits","behavioral_candidates","churn_seasonality","e012_robust"]
tabs={}
for nm in names:
    d=L(nm)
    if d is not None:
        tabs[nm]=d
        ok=('household_key' in d.columns) and ('snapshot_day' in d.columns)
        print(f"{nm:26s} {str(d.shape):14s} keys_ok={ok}")

e009=tabs["e009_ewma_longlags"]
cols=list(e009.columns)
print("\nE009 ncols", len(cols))
print("E009 dtypes:", e009.dtypes.value_counts().to_dict())
print("object cols:", [c for c in cols if e009[c].dtype==object])
print("COLS:", "|".join(map(str,cols)))
b=agent_api.baseline_features()
print("baseline", b.shape)
