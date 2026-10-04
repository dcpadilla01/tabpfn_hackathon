import agent_api as A
import pandas as pd, numpy as np

tgt = A.load_saved('my_targets.parquet')
df = A.load_saved('e009_macro.parquet')
d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
trm = d[d.snapshot_day.isin([95,123,151,179,207,235,263,291,319])]
vam = d[d.snapshot_day.isin([347,375,403,431])]

num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
print("num:", len(num), "cat:", cat)

def prep(df):
    Xn = df[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(df[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=df.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)

Xtr, Xva = prep(trm), prep(vam)
Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
ytr, yva = trm.y.values, vam.y.values
A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
D = np.eye(A_.shape[1]); D[-1,-1]=0
for lam in [1,10,30,100]:
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    pv = np.clip(Av@w, 0, None)
    print(f"lam={lam}: pv mean {pv.mean():.1f} std {pv.std():.1f} | MAE {np.abs(pv-yva).mean():.2f} | baseline MAE(median {np.median(ytr):.0f}) {np.abs(np.median(ytr)-yva).mean():.2f}")
# also check: does Xtr row order match ytr? spot check first household
print(trm[['household_key','snapshot_day','y']].head(3))
print(Xtr.iloc[:3,:4])