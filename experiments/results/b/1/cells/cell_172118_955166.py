import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
TR_SNAPS = [95,123,151,179,207,235,263,291,319]
VA_SNAPS = [347,375,403,431]

def prep(df):
    num, cat = [], []
    for c in df.columns:
        if c in ('household_key','snapshot_day'): continue
        if df[c].dtype.kind in 'ifb': num.append(c)
        else: cat.append(c)
    Xn = df[num].astype(float)
    Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(df[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=df.index)
    X = pd.concat([Xn, Xc.astype(float)], axis=1)
    return X

def ridge_eval(path, lam=30.0, logt=False):
    df = A.load_saved(path)
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR_SNAPS)]; vam = d[d.snapshot_day.isin(VA_SNAPS)]
    Xtr, Xva = prep(trm), prep(vam)
    Xtr, Xva = Xtr.align(Xva, join='left', fill_value=0.0)
    ytr = trm.y.values; yva = vam.y.values
    ytr_f, yva_f = (np.log1p(ytr), np.log1p(yva)) if logt else (ytr, yva)
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1] = 0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr_f)
    pv = Av@w
    if logt: pv = np.expm1(pv)
    pv = np.clip(pv, 0, None)
    return np.abs(pv - yva).mean(), Xtr.shape[1]

for path in ['e009_macro.parquet','e001_history.parquet','e013_union.parquet','e014_base.parquet','e015_base.parquet','e017_gapfill.parquet','e012_full.parquet']:
    m1 = ridge_eval(path, 30.0, False)
    m2 = ridge_eval(path, 30.0, True)
    print(f"{path:24s} ridge raw {m1[0]:7.3f} ({m1[1]:4d}f) | ridge log {m2[0]:7.3f}")