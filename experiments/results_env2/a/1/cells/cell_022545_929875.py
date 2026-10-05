import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved

F = load_saved("allF.parquet"); C = load_saved("camp_feats.parquet")
F2 = F.merge(C, on=["household_key","snapshot_day"], how="left").fillna({"cmp_laststart":9999.0})
feats = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
feats2 = feats + ["cmp_active","cmp_start28","cmp_endfut","cmp_fut_days","cmp_laststart","cmp_act_A","cmp_act_B","cmp_act_C"]
tr = F2[F2.future_spend_4w.notna()].copy()
TRN, HOLD = tr[tr.snapshot_day<=375], tr[tr.snapshot_day.isin([403,431])]
yv = HOLD.future_spend_4w.values

def fit_pred(cols, df_tr, df_ho, half=140.0, rounds=1200, seed=1):
    w = 0.5**((df_tr.snapshot_day.values-df_tr.snapshot_day.max())/half)
    p = dict(objective="reg:quantileerror", quantile_alpha=0.5, eta=0.03, max_depth=6,
             min_child_weight=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist", seed=seed)
    b = xgb.train(p, xgb.DMatrix(df_tr[cols], label=df_tr.future_spend_4w.values, weight=w), num_boost_round=rounds)
    return np.clip(b.predict(xgb.DMatrix(df_ho[cols])),0,None)

t0=time.time()
p1 = fit_pred(feats, TRN, HOLD)
p2 = fit_pred(feats2, TRN, HOLD)
print("base MAE %.3f | +camp MAE %.3f  (%.0fs)" % (np.abs(p1-yv).mean(), np.abs(p2-yv).mean(), time.time()-t0))
