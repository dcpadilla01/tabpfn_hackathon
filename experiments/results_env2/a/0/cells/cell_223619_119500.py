import agent_api as A, pandas as pd, numpy as np, time, warnings
warnings.filterwarnings("ignore")
import xgboost as xgb

f = A.load_saved("feats_v3.parquet").drop(columns=["index"])
tt = A.train_targets()
df = tt.merge(f, on=["household_key","snapshot_day"], how="inner")
FEATS = [c for c in f.columns if c not in ("household_key","snapshot_day")]
BASE = dict(n_estimators=800, learning_rate=0.05, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, reg_lambda=1.0, n_jobs=4,
        objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist")

def fit(train_days, params=None, sample_weight=None):
    tr = df[df.snapshot_day.isin(train_days)]
    p = dict(BASE)
    if params: p.update(params)
    m = xgb.XGBRegressor(**p)
    w = sample_weight(tr) if sample_weight else None
    m.fit(tr[FEATS], tr.future_spend_4w, sample_weight=w)
    return m

def evaluate(mut_name, params=None, sample_weight=None, blend_fn=None):
    t0=time.time(); out={}
    for held in [403, 431]:
        td = [d for d in range(95,432,28) if d < held]
        m = fit(td, params, sample_weight)
        pr = df[df.snapshot_day==held]
        pred = m.predict(pr[FEATS])
        if blend_fn is not None:
            pred = blend_fn(pred, pr)
        out[held] = np.abs(pred - pr.future_spend_4w.values).mean()
    print(f"{mut_name:42s} 403:{out[403]:7.3f} 431:{out[431]:7.3f} avg:{np.mean(list(out.values())):7.3f} ({time.time()-t0:.0f}s)")
    return out

# 0) repro
evaluate("repro E005")
# 1) blend with exp4w_blend persistence
for w in [0.05, 0.1, 0.2]:
    evaluate(f"blend model + {w}*exp4w_blend", blend_fn=lambda p, pr, w=w: (1-w)*p + w*pr.exp4w_blend.values)
# 2) recency-weighted loss
evaluate("sample_weight exp(-age/112)", sample_weight=lambda tr: np.exp(-(431-tr.snapshot_day)/112))
evaluate("sample_weight exp(-age/56)", sample_weight=lambda tr: np.exp(-(431-tr.snapshot_day)/56))
