
import pandas as pd, numpy as np, agent_api, xgboost as xgb

feats = agent_api.load_saved('feats_v3.parquet')
tt = agent_api.train_targets()
df = tt.merge(feats, on=['household_key','snapshot_day'], how='left')
FE = [c for c in feats.columns if c not in ('index','household_key','snapshot_day')]

def make_xy(d):
    X = d[FE].copy()
    for c in X.columns: X[c] = pd.to_numeric(X[c], errors='coerce')
    return X, d['future_spend_4w'].values

def qmodel(Xtr,ytr,alpha=0.5,depth=4,mcw=20,nest=600,lr=0.05,seed=0):
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, max_depth=depth,
                         min_child_weight=mcw, n_estimators=nest, learning_rate=lr,
                         subsample=0.8, colsample_bytree=0.8, n_jobs=4, random_state=seed)
    m.fit(Xtr,ytr); return m

def eval_config(name, mode, w, alphas=(0.5,), seeds=(0,)):
    maes = {}
    for hd in [403, 431]:
        tr = df[df.snapshot_day < hd]; va = df[df.snapshot_day == hd]
        Xtr,ytr = make_xy(tr); Xv,yv = make_xy(va)
        blv = Xv['exp4w_blend'].values
        ps = []
        for sd in seeds:
            for a in alphas:
                if mode=='direct':
                    ps.append(qmodel(Xtr,ytr,alpha=a,seed=sd).predict(Xv))
                else:  # resid
                    ps.append(qmodel(Xtr,ytr-blv_tr_blenda if False else ytr-Xtr['exp4w_blend'].values,seed=sd).predict(Xv))
        p = np.mean(ps,axis=0)
        f = np.clip(w*p+(1-w)*blv,0,None) if mode=='direct' else np.clip(p+blv,0,None)
        maes[hd] = round(np.abs(f-yv).mean(),3)
    print(f"{name:28s} 403:{maes[403]:7.3f}  431:{maes[431]:7.3f}  avg:{np.mean(list(maes.values())):7.3f}")
    return np.mean(list(maes.values()))

eval_config("A E007 exact (w.7,s0)", 'direct', 0.7)
eval_config("B ens3 seeds w.7", 'direct', 0.7, seeds=(0,1,2))
eval_config("C ens3 seeds w.6", 'direct', 0.6, seeds=(0,1,2))
eval_config("D resid-boost ens3", 'resid', 0.0, seeds=(0,1,2))
eval_config("E ens3 alpha(.45,.5) w.7", 'direct', 0.7, alphas=(0.45,0.5), seeds=(0,1))
