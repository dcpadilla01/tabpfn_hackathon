import agent_api as A
import pandas as pd, numpy as np

def cand_fn(view, snapshot_day):
    d = int(snapshot_day)
    hhs = pd.Index(view.households)
    t = view.transactions
    t = t[t.household_key.isin(hhs)]
    if len(t) == 0:
        return pd.DataFrame(index=hhs)
    # window index k: 1 = (d-28, d], 2 = (d-56, d-28], ...
    k = ((d - t.day - 1) // 28 + 1).astype(int)
    t = t.assign(k=k)
    t = t[(t.k >= 1) & (t.k <= 13)]
    g = t.groupby(['household_key','k'])
    spend = g.sales_value.sum().unstack(fill_value=0.0)
    nb = g.basket_id.nunique().unstack(fill_value=0.0)
    K = range(1, 14)
    spend = spend.reindex(columns=K, fill_value=0.0).reindex(hhs, fill_value=0.0)
    nb = nb.reindex(columns=K, fill_value=0.0).reindex(hhs, fill_value=0.0)
    out = pd.DataFrame(index=hhs)
    for kk in K:
        out[f'a{kk}'] = spend[kk]
        out[f'al{kk}'] = np.log1p(spend[kk])
        out[f'nw{kk}'] = nb[kk]
        out[f'aw{kk}'] = (spend[kk] > 0).astype(float)
    arr = spend.values
    out['z6'] = (arr[:, :6] == 0).sum(1)
    out['z13'] = (arr == 0).sum(1)
    out['med13'] = np.median(arr, axis=1)
    nz = np.where(arr > 0, arr, np.nan)
    with np.errstate(all='ignore'):
        out['mednz13'] = np.nanmedian(nz, axis=1)
        out['q75_13'] = np.nanpercentile(arr, 75, axis=1)
        out['amax13'] = np.nanmax(arr, axis=1)
    out['ratio_23'] = out.a2 / (out.a3 + 1.0)
    out['ratio_2m'] = out.a2 / (out.med13 + 1.0)
    # median-shaped composites (need E015 cols; merge later outside fn) -> placeholder none
    return out

tab = A.build_features(cand_fn)
print(tab.shape)
p = A.save_table(tab, 'e019_cand')
print("saved", p)

# ---- proxy eval ----
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
cand = A.load_saved('e019_cand')
print("e015 proxy:", round(ridge_eval(base),3))
fams = {
 'grid_spend': [f'a{k}' for k in range(1,14)] + [f'al{k}' for k in range(1,14)],
 'grid_cnt': [f'nw{k}' for k in range(1,14)] + [f'aw{k}' for k in range(1,14)],
 'stats': ['z6','z13','med13','mednz13','q75_13','amax13','ratio_23','ratio_2m'],
}
for name, cols in fams.items():
    df = pd.concat([base, cand[cols]], axis=1)
    print(f"e015+{name:10s}: {ridge_eval(df):.3f}")
df = pd.concat([base, cand[[c for cols in fams.values() for c in cols]]], axis=1)
print("e015+ALLCAND:", round(ridge_eval(df),3), df.shape)