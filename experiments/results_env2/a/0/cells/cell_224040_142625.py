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

def evaluate(name, params=None, blend_fn=None, obj="reg:quantileerror"):
    t0=time.time(); out={}
    for held in [403, 431]:
        td = [d for d in range(95,432,28) if d < held]
        tr = df[df.snapshot_day.isin(td)]
        p = dict(BASE); p["objective"]=obj
        if params: p.update(params)
        m = xgb.XGBRegressor(**p).fit(tr[FEATS], tr.future_spend_4w)
        pr = df[df.snapshot_day==held]
        pred = m.predict(pr[FEATS])
        if blend_fn is not None: pred = blend_fn(pred, pr)
        out[held] = np.abs(pred - pr.future_spend_4w.values).mean()
    print(f"{name:44s} 403:{out[403]:7.3f} 431:{out[431]:7.3f} avg:{np.mean(list(out.values())):7.3f} ({time.time()-t0:.0f}s)")
    return out

def blend_w(w):
    return lambda p, pr: (1-w)*p + w*pr.exp4w_blend.values

# combos
evaluate("blend .3 + depth4", params=dict(max_depth=4), blend_fn=blend_w(0.3))
evaluate("blend .3 + depth4 + mcw20", params=dict(max_depth=4, min_child_weight=20), blend_fn=blend_w(0.3))
evaluate("blend .35 depth4", params=dict(max_depth=4), blend_fn=blend_w(0.35))
# ensemble quantile + pseudohuber averaged, then blend
def ens_blend(w, wh=0.5):
    return lambda p, pr: (1-w)*((1-wh)*p + wh*pr.pred_huber) + w*pr.exp4w_blend.values
for held in [403, 431]:
    td = [d for d in range(95,432,28) if d < held]
    tr = df[df.snapshot_day.isin(td)]
    pr = df[df.snapshot_day==held]
    m1 = xgb.XGBRegressor(**{**BASE, "objective":"reg:quantileerror"}).fit(tr[FEATS], tr.future_spend_4w)
    m2 = xgb.XGBRegressor(**{**BASE, "objective":"reg:pseudohubererror"}).fit(tr[FEATS], tr.future_spend_4w)
    pr = pr.assign(pred_huber=m2.predict(pr[FEATS]))
    for wh in [0.3, 0.5]:
        for w in [0.2, 0.3]:
            pred = (1-w)*((1-wh)*m1.predict(pr[FEATS]) + wh*pr.pred_huber) + w*pr.exp4w_blend.values
            print(f"  held {held} ens huber_w{wh} blend{w}: {np.abs(pred-pr.future_spend_4w.values).mean():.3f}")
