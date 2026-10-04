import agent_api as A
import pandas as pd, numpy as np
tgt = A.load_saved('my_targets.parquet')
TR = [95,123,151,179,207,235,263,291,319]; VA = [347,375,403,431]
def prep(d):
    num = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind in 'ifb']
    cat = [c for c in d.columns if c not in ('household_key','snapshot_day','y') and d[c].dtype.kind not in 'ifb']
    Xn = d[num].astype(float); Xn = Xn.fillna(Xn.median())
    mu, sd = Xn.mean(), Xn.std().replace(0,1)
    Xn = ((Xn-mu)/sd).fillna(0.0).clip(-5,5)
    Xc = pd.get_dummies(d[cat].astype(str), dummy_na=True) if cat else pd.DataFrame(index=d.index)
    return pd.concat([Xn, Xc.astype(float)], axis=1)
def ridge_eval(df, lam=100.0):
    d = df.merge(tgt, on=['household_key','snapshot_day'], how='left')
    trm = d[d.snapshot_day.isin(TR)]; vam = d[d.snapshot_day.isin(VA)]
    Xtr, Xva = prep(trm), prep(vam)
    Xva = Xva.reindex(columns=Xtr.columns, fill_value=0.0)
    ytr, yva = trm.y.values, vam.y.values
    A_ = np.hstack([Xtr.values, np.ones((len(Xtr),1))])
    Av = np.hstack([Xva.values, np.ones((len(Xva),1))])
    D = np.eye(A_.shape[1]); D[-1,-1]=0
    w = np.linalg.solve(A_.T@A_ + lam*D, A_.T@ytr)
    return np.abs(np.clip(Av@w,0,None) - yva).mean()

base = A.load_saved('e015_base.parquet')
c01 = lambda x: np.clip(x, 0, 1)
p13, p6, p3 = base.p13, base.p6, base.p3
u13, u6, u3 = base.usual13, base.usual6, base.usual3
e13, e6 = base.e13, base.e6
ly = base.ly_spend.fillna(0)
M = pd.DataFrame({
 'med13a': u13*c01((p13-0.4)/0.2), 'med13b': u13*c01(2*p13-1),
 'med6b': u6*c01(2*p6-1), 'med3b': u3*c01(2*p3-1),
 'med13_sq': u13*p13*p13, 'med_ly': np.maximum(u13*c01(2*p13-1), 0.7*ly),
 'med13_4b': base.usual13_4*c01(2*base.p13_4-1),
 'med_geo': np.sqrt(np.maximum(e13,0)*u13)*c01(2*p13-1),
 'e13_p13': e13*p13, 'u13_p13': u13*p13,
 'e13_slope': e13*base.slope6.fillna(0), 'p13_ten': p13*np.log1p(base.tenure),
 'e13_wmax': e13*base.w_max7.fillna(0)}).fillna(0.0)
df = pd.concat([base, M], axis=1)
for lam in [50, 100, 200]:
    print(f"e015+MEDCOMBO lam={lam}: {ridge_eval(df, lam):.3f}")
p = A.save_table(df, 'e019_medcombo')
print("saved:", p, df.shape)