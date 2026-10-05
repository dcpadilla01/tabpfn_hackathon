
import agent_api, pandas as pd, numpy as np, xgboost as xgb, time
F = agent_api.load_saved('e004_features.parquet'); T = agent_api.train_targets()
D = F.merge(T, on=['household_key','snapshot_day'], how='left')
oldf = [c for c in F.columns if c not in ('household_key','snapshot_day')]
base = dict(n_estimators=2400, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=20, reg_lambda=1.0,
            tree_method='hist', n_jobs=8, objective='reg:quantileerror', quantile_alpha=0.5)

def preds(V, params, decay):
    tr = D[D.snapshot_day < V]; va = D[D.snapshot_day == V]
    w = 0.5 ** ((V - tr.snapshot_day) / decay)
    m = xgb.XGBRegressor(**params); m.fit(tr[feats:=oldf], tr.future_spend_4w, sample_weight=w, verbose=False)
    return np.clip(m.predict(va[oldf]), 0, None), va

t0=time.time()
for V in (431, 403):
    P=[]
    p = dict(base); p['objective']='reg:pseudohubererror'; p['huber_slope']=5
    ph,_ = preds(V, p, 140)
    print(f'V={V} huber5 d140: {np.abs(ph-va_y if False else 0):.3f}' if False else '', end='')
    va = D[D.snapshot_day==V]; y = va.future_spend_4w.values
    print(f'V={V} huber5 d140: {np.abs(ph-y).mean():.3f}')
    for dec in (140, 200):
        pv,_ = preds(V, base, dec)
        print(f'V={V} quant d{dec}: {np.abs(pv-y).mean():.3f}')
        P.append(pv)
    ens = np.mean(P, axis=0)
    print(f'V={V} ens(140,200): {np.abs(ens-y).mean():.3f}')
print(f'({time.time()-t0:.0f}s)')
