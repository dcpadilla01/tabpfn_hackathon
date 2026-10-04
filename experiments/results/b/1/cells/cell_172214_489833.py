import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
TR_SNAPS = [95,123,151,179,207,235,263,291,319]
VA_SNAPS = [347,375,403,431]

def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)

def ridge_eval(path, lam=30.0):
    df = A.load_saved(path)
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR_SNAPS)]; vam = d[d.snapshot_day.isin(VA_SNAPS)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    pv = np.clip(Av@w, 0, None)
    return np.abs(pv - yva).mean(), Xtr.shape[1]

for path in ['e015_base.parquet','e017_gapfill.parquet','e017r.parquet']:
    for lam in [30, 100, 300]:
        m, nf = ridge_eval(path, lam)
        print(f"{path:22s} lam={lam:4d} MAE {m:6.3f} ({nf}f)")
    print()