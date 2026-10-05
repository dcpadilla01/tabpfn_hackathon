import pandas as pd, numpy as np, xgboost as xgb, time, warnings
warnings.filterwarnings("ignore")
from agent_api import load_saved

F = load_saved("allF.parquet")
W = load_saved("f_weekly.parquet")
print(W.columns.tolist())
# columns already in allF or tested before
skip = {"household_key","snapshot_day","lag28_56","lag56_84","lag84_112","sp28_yag","sp84_yag"}
newcols = [c for c in W.columns if c not in skip]
print("new cols:", newcols)
F2 = F.merge(W[["household_key","snapshot_day"]+newcols], on=["household_key","snapshot_day"], how="left")
print(F2.shape, "NaN new:", F2[newcols].isna().sum().sum())

tr = F2[F2.future_spend_4w.notna()].copy()
TRN = tr[tr.snapshot_day <= 375]
HOLD = tr[tr.snapshot_day.isin([403,431])]
yv = HOLD.future_spend_4w.values

def fit_pred(cols, df_tr, df_ho, kind="q", rounds=1200, lr=0.03, half=140.0, seed=1, depth=6, mcw=5):
    w = 0.5 ** ((df_tr.snapshot_day.values - df_tr.snapshot_day.max()) / half)
    if kind=="q": lab, obj = df_tr.future_spend_4w.values, dict(objective="reg:quantileerror", quantile_alpha=0.5)
    elif kind=="l1": lab, obj = df_tr.future_spend_4w.values, dict(objective="reg:absoluteerror")
    elif kind=="sqrt": lab, obj = np.sqrt(df_tr.future_spend_4w.values), dict(objective="reg:squarederror")
    p = dict(eta=lr, max_depth=depth, min_child_weight=mcw, subsample=0.8, colsample_bytree=0.8,
             tree_method="hist", seed=seed); p.update(obj)
    b = xgb.train(p, xgb.DMatrix(df_tr[cols], label=lab, weight=w), num_boost_round=rounds)
    pr = b.predict(xgb.DMatrix(df_ho[cols]))
    if kind=="sqrt": pr = np.square(pr)
    return np.clip(pr, 0, None)

base_cols = [c for c in F.columns if c not in ("household_key","snapshot_day","future_spend_4w")]
new_cols = base_cols + newcols
t0=time.time()
for name, cols, kw in [
    ("base q",        base_cols, dict(kind="q")),
    ("base+weekly q", new_cols,  dict(kind="q")),
    ("base l1",       base_cols, dict(kind="l1")),
    ("base+weekly l1",new_cols,  dict(kind="l1")),
    ("base sqrt",     base_cols, dict(kind="sqrt")),
]:
    p = fit_pred(cols, TRN, HOLD, **kw)
    print("%-16s MAE %.3f mean %.1f" % (name, np.abs(p-yv).mean(), p.mean()))
print("%.0fs" % (time.time()-t0))
