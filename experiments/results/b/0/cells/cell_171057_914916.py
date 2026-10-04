import agent_api, pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

df = agent_api.load_saved('e018_basestab.parquet')
tt = agent_api.train_targets()
df = df.merge(tt, on=['household_key','snapshot_day'], how='left')

def prep(frame):
    feats = [c for c in frame.columns if c not in ('household_key','snapshot_day','future_spend_4w')]
    num = frame[feats].apply(pd.to_numeric, errors='coerce').replace([np.inf,-np.inf], np.nan)
    num = num.drop(columns=[c for c in num.columns if num[c].notna().sum()==0])
    return num.fillna(num.median())

def eval_table(frame, label):
    X = prep(frame); y = frame.future_spend_4w.values
    is_tr = frame.future_spend_4w.notna().values
    mu = X[is_tr].mean(); sd = X[is_tr].std().replace(0,1)
    Z = ((X-mu)/sd).clip(-5,5); Z['ic']=1.0; Zv=Z.values
    def fit(m, lam=5.0):
        Zt=Zv[m]; A=Zt.T@Zt+lam*np.eye(Zt.shape[1]); return np.linalg.solve(A, Zt.T@y[m])
    tot=[]
    for d in [263,291,319,347,375,403,431]:
        trm = is_tr & (frame.snapshot_day<d).values; vam = is_tr & (frame.snapshot_day==d).values
        w=fit(trm); r = y[vam]-Zv[vam]@w
        tot.append((d, r.mean(), np.abs(r).mean()))
    arr = np.array([t[2] for t in tot])
    print(f"{label}: meanOOF MAE={arr.mean():.2f} | " + " ".join(f"d{int(a)}:b{b:+.0f}/m{c:.0f}" for a,b,c in tot))
    return Zv, is_tr, y

# A) seasonality: week-of-year sin/cos at FUTURE window midpoint
d = df.copy()
mid = d.snapshot_day + 14
wk = (mid+8)//7
d['fut_wk_sin'] = np.sin(2*np.pi*wk/52.18)
d['fut_wk_cos'] = np.cos(2*np.pi*wk/52.18)
d['fut_wk'] = wk
eval_table(d, 'e018 + future-week sin/cos')

# B) seasonality: retailer-wide future-window spend per active household, computed from txn history (leakage-safe: uses only past txn)
v = agent_api.snapshot(459)
txn = v.transactions
tot = txn.groupby('day').sales_value.sum()
d['fut_win_daily'] = [tot.iloc[min(int(t)+1, len(tot)-1):min(int(t)+28, len(tot)-1)].sum() for t in d.snapshot_day]
eval_table(d, 'e018 + future-window retailer daily total (diag only)')

# C) day-level target encoding of snapshot_day (strictly prior days only)
d = df.copy()
day_stats = df[df.future_spend_4w.notna()].groupby('snapshot_day').future_spend_4w.agg(['mean','median','count'])
gm = df[df.future_spend_4w.notna()].future_spend_4w.mean()
d['te_day_mean'] = [day_stats.loc[day_stats.index<day, 'mean'].mean() if (day_stats.index<day).any() else gm for day in d.snapshot_day]
d['te_day_last'] = [day_stats.loc[day_stats.index<day, 'mean'].iloc[-1] if (day_stats.index<day).any() else gm for day in d.snapshot_day]
d['te_day_med'] = [day_stats.loc[day_stats.index<day, 'median'].mean() if (day_stats.index<day).any() else gm for day in d.snapshot_day]
eval_table(d, 'e018 + prior-day TE (mean/last/median)')