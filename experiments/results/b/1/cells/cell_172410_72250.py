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
cand = A.load_saved('e019_cand.parquet')
cand = cand.rename(columns={c: 'g_'+c for c in cand.columns if c not in ('household_key','snapshot_day')})
fams = {
 'grid_spend': [f'g_a{k}' for k in range(1,14)] + [f'g_al{k}' for k in range(1,14)],
 'grid_cnt': [f'g_nw{k}' for k in range(1,14)] + [f'g_aw{k}' for k in range(1,14)],
 'stats': ['g_z6','g_z13','g_med13','g_mednz13','g_q75_13','g_amax13','g_ratio_23','g_ratio_2m'],
}
print("e015 proxy:", round(ridge_eval(base),3))
for name, cols in fams.items():
    print(f"e015+{name:10s}: {ridge_eval(pd.concat([base, cand[cols]], axis=1)):.3f}")
allc = [c for cols in fams.values() for c in cols]
df = pd.concat([base, cand[allc]], axis=1)
print("e015+ALLCAND:", round(ridge_eval(df),3), df.shape)
p = A.save_table(df, 'e019_final')
print("saved:", p)