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

def evaluate(name, params=None, blend_fn=None, n_est=800, lr=0.05):
    t0=time.time(); out={}
    for held in [403, 431]:
        td = [d for d in range(95,432,28) if d < held]
        tr = df[df.snapshot_day.isin(td)]
        p = dict(BASE); p["n_estimators"]=n_est; p["learning_rate"]=lr
        if params: p.update(params)
        m = xgb.XGBRegressor(**p).fit(tr[FEATS], tr.future_spend_4w)
        pr = df[df.snapshot_day==held]
        pred = m.predict(pr[FEATS])
        if blend_fn is not None: pred = blend_fn(pred, pr)
        out[held] = np.abs(pred - pr.future_spend_4w.values).mean()
    print(f"{name:44s} 403:{out[403]:7.3f} 431:{out[431]:7.3f} avg:{np.mean(list(out.values())):7.3f} ({time.time()-t0:.0f}s)")
    return out

# blend sweep
for w in [0.3, 0.4, 0.5]:
    evaluate(f"blend + {w}*exp4w_blend", blend_fn=lambda p, pr, w=w: (1-w)*p + w*pr.exp4w_blend.values)
# blend with lag1 instead
evaluate("blend + 0.2*spend28_lag1", blend_fn=lambda p, pr: 0.8*p + 0.2*pr.spend28_lag1.values)
evaluate("blend + 0.2*exp4w_all", blend_fn=lambda p, pr: 0.8*p + 0.2*pr.exp4w_all.values)
# model params
evaluate("depth 8", params=dict(max_depth=8))
evaluate("depth 4", params=dict(max_depth=4))
evaluate("lr .03 n1600", n_est=1600, lr=0.03)
evaluate("mcw 1", params=dict(min_child_weight=1))
evaluate("mcw 20", params=dict(min_child_weight=20))
evaluate("subsample .6 col .5", params=dict(subsample=0.6, colsample_bytree=0.5))
evaluate("reg_lambda 5", params=dict(reg_lambda=5.0))
