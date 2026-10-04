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

def ridge_eval(df, lam=100.0):
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
    return np.abs(pv - yva).mean()

base = A.load_saved('e015_base.parquet')
print("e015 proxy MAE:", round(ridge_eval(base),3))

def clip01(x): return np.clip(x, 0, 1)
p13, p6, p3 = base.p13, base.p6, base.p3
u13, u6, u3 = base.usual13, base.usual6, base.usual3
e13, e6 = base.e13, base.e6
ly = base.ly_spend.fillna(0)

V = {}
V['medA'] = pd.DataFrame({
    'med13a': u13*clip01((p13-0.4)/0.2), 'med13b': u13*clip01(2*p13-1),
    'med6b': u6*clip01(2*p6-1), 'med3b': u3*clip01(2*p3-1),
    'med13_sq': u13*p13*p13, 'med_ly': np.maximum(u13*clip01(2*p13-1), 0.7*ly)})
V['medB'] = pd.DataFrame({
    'med13b': u13*clip01(2*p13-1), 'med6b': u6*clip01(2*p6-1),
    'med13_4b': base.usual13_4*clip01(2*base.p13_4-1),
    'med_ly_max': np.maximum(u13*clip01(2*p13-1), 0.7*ly),
    'med_geo': np.sqrt(np.maximum(e13,0)*u13)*clip01(2*p13-1)})
V['inter'] = pd.DataFrame({
    'e13_p13': e13*p13, 'u13_p13': u13*p13, 'e13_slope': e13*base.slope6.fillna(0),
    'p13_ten': p13*np.log1p(base.tenure), 'e13_wmax': e13*base.w_max7.fillna(0)})
V['rank'] = pd.DataFrame({
    'rk_e13': base.groupby('snapshot_day').e13.rank(pct=True),
    'rk_p13': base.groupby('snapshot_day').p13.rank(pct=True),
    'rk_u13': base.groupby('snapshot_day').u13.rank(pct=True) if 'u13' in base else base.groupby('snapshot_day').usual13.rank(pct=True),
    'rk_s28': base.groupby('snapshot_day').log_spend28.rank(pct=True)})
for k, v in V.items():
    v = v.fillna(0.0)
    df = pd.concat([base, v], axis=1)
    print(f"e015+{k:6s} proxy MAE: {ridge_eval(df):.3f}  (+{v.shape[1]}f)")

df_all = pd.concat([base] + [v.fillna(0.0) for v in V.values()], axis=1)
print("e015+ALL proxy MAE:", round(ridge_eval(df_all),3), df_all.shape)