import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
print("shape", t.shape)
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
print(len(feats), "features")
print("dtypes:", t.dtypes.value_counts().to_dict())
print("cols:", feats)

tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("merged", df.shape)
print("snap days:", sorted(df.snapshot_day.unique()))
print("rows per snap:", t.groupby('snapshot_day').size().to_dict())

train = df[df.snapshot_day <= 431]
ytr = train['future_spend_4w'].astype(float)
print("target stats:", ytr.describe().round(2).to_dict())

# baseline references on pseudo-val snap 431
pv = train[train.snapshot_day == 431]
print("MAE global median:", round(np.abs(pv.future_spend_4w - ytr.median()).mean(), 3))
for c in feats:
    if '28' in c and t[c].dtype != object and train[c].notna().mean() > 0.9:
        pass

num_cols = [c for c in feats if str(t[c].dtype) not in ('object','category')]
cat_cols = [c for c in feats if str(t[c].dtype) in ('object','category')]
print("num", len(num_cols), "cat", len(cat_cols), cat_cols)

# univariate |corr| with target
cor = {}
for c in num_cols:
    x = train[c].astype(float); ok = x.notna()
    if ok.sum() > 50 and x[ok].std() > 0:
        cor[c] = abs(np.corrcoef(x[ok], ytr[ok])[0,1])
cor = pd.Series(cor).sort_values()
print("\nweakest 25 by |corr|:"); print(cor.head(25).round(4))

# NaN rates overall and at validation snaps
nanr = t[feats].isna().mean()
print("\ntop NaN rates:"); print(nanr.sort_values(ascending=False).head(8).round(3))
val = t[t.snapshot_day >= 459]
print("NaN rates at val snaps (max):", val[feats].isna().mean().max().round(3))

# drift: trend of per-snapshot means across ALL snaps incl validation
dr = {}
for c in num_cols:
    m = t.groupby('snapshot_day')[c].mean().dropna()
    if len(m) >= 10 and m.std() > 0:
        dr[c] = abs(np.corrcoef(m.index.astype(float), m.values)[0,1])
dr = pd.Series(dr).sort_values(ascending=False)
print("\nmost drifting 15 (mean vs snapshot_day):"); print(dr.head(15).round(3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')

# target mean per train snapshot
print("target mean/median per snap:")
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','mean']).round(1).T)

e11 = agent_api.load_saved('e011_pruned_basket.parquet')
print("\ne011 shape", e11.shape)
# tenure-like cols in e011
cands = [c for c in e11.columns if any(k in c for k in ('tenure','first','days_since'))]
print("tenure-ish cols:", cands)

m = e11.merge(tt, on=['household_key','snapshot_day'], how='inner')
g = m.groupby('snapshot_day')
show = ['tenure','days_since_first','days_since_last','spend_364','trips_364','weekly_mean_12','dec_112','z_dec_224','spend_28','future_spend_4w']
show = [c for c in show if c in m.columns]
print("\nper-snapshot means (drift check):")
print(g[show].mean().round(2))

# how much of spend_364 drift is tenure? corr of feature with tenure, and partial
tr = m[m.snapshot_day<=431]
for c in ['spend_364','trips_364','dec_112','weekly_mean_12','z_dec_224','days_since_last']:
    x = tr[c]; tn = tr['tenure']
    ok = x.notna()&tn.notna()
    print(c, "corr w/ tenure:", round(np.corrcoef(x[ok],tn[ok])[0,1],3))


# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num = [c for c in feats if str(t[c].dtype) not in ('object','category')]

train = t[t.snapshot_day <= 431]
val = t[t.snapshot_day >= 459]

rows = []
for c in num:
    x_tr = train[c]; x_va = val[c]
    gm = t.groupby('snapshot_day')[c].mean().dropna()
    drift = abs(np.corrcoef(gm.index.astype(float), gm.values)[0,1]) if len(gm)>=10 and gm.std()>0 else np.nan
    # val mean vs late-train mean shift, scaled by pooled std
    sd = t[c].std()
    shift = abs(x_va.mean() - x_tr[x_tr.snapshot_day>=375].mean())/sd if sd>0 else np.nan
    # univariate corr w target
    dfm = t.merge(tt, on=['household_key','snapshot_day'])
    tr = dfm[dfm.snapshot_day<=431]
    ok = tr[c].notna()
    cor = abs(np.corrcoef(tr.loc[ok,c].astype(float), tr.loc[ok,'future_spend_4w'])[0,1]) if ok.sum()>50 and tr.loc[ok,c].std()>0 else 0
    rows.append((c, cor, drift if drift==drift else -1, shift if shift==shift else -1, t[c].isna().mean()))

res = pd.DataFrame(rows, columns=['feat','abs_corr','drift','valshift','nan']).sort_values('drift', ascending=False)
print("Top 20 drifters (drift | corr | val-shift | nan):")
print(res.head(20).round(3).to_string())
print("\nLow-corr features (abs_corr<0.06):")
print(res[res.abs_corr<0.06].round(3).to_string())
print("\nVal shift top:")
print(res.sort_values('valshift',ascending=False).head(12).round(3).to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e013_stationary.parquet')
tt = agent_api.train_targets()
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
num = [c for c in feats if str(t[c].dtype) not in ('object','category')]

train = t[t.snapshot_day <= 431]
val = t[t.snapshot_day >= 459]
late = train[train.snapshot_day >= 375]
dfm = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
tr = dfm[dfm.snapshot_day <= 431]

rows = []
for c in num:
    gm = t.groupby('snapshot_day')[c].mean().dropna()
    drift = abs(np.corrcoef(gm.index.astype(float), gm.values)[0,1]) if len(gm)>=10 and gm.std()>0 else -1
    sd = t[c].std()
    shift = abs(val[c].mean() - late[c].mean())/sd if sd and sd>0 else -1
    ok = tr[c].notna()
    cor = abs(np.corrcoef(tr.loc[ok,c].astype(float), tr.loc[ok,'future_spend_4w'])[0,1]) if ok.sum()>50 and tr.loc[ok,c].std()>0 else 0
    rows.append((c, cor, drift, shift, t[c].isna().mean()))

res = pd.DataFrame(rows, columns=['feat','abs_corr','drift','valshift','nan']).sort_values('drift', ascending=False)
print("Top 20 drifters:")
print(res.head(20).round(3).to_string())
print("\nLow-corr (abs_corr<0.06):")
print(res[res.abs_corr<0.06].round(3).to_string())
print("\nTop val-shift:")
print(res.sort_values('valshift',ascending=False).head(12).round(3).to_string())


# ---- cell ----
import agent_api, pandas as pd, numpy as np

def fn(view, s):
    tx = view.table('transactions')
    hh = pd.Index(view.households, name='household_key')
    first = tx.groupby('household_key')['day'].min().reindex(hh)
    tenure = (s - first).astype(float)
    out = pd.DataFrame(index=hh)
    out['tenure_cap364'] = tenure.clip(upper=364)
    eff = tenure.clip(lower=1.0)
    for W in (112, 182, 364):
        w = tx[tx.day > s - W]
        sp = w.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
        out[f'rate_{W}'] = sp / eff.clip(upper=W) * 7.0   # weekly-equivalent spend rate
    tr = tx[tx.day > s - 364].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0.0)
    out['trips_rate_364'] = tr / (eff.clip(upper=364) / 7.0)
    age = (s - tx['day']).astype(float)
    dec = (tx['sales_value'] * (0.5 ** (age / 112.0))).groupby(tx['household_key']).sum().reindex(hh).fillna(0.0)
    normw = (112.0 / np.log(2.0)) * (1.0 - 0.5 ** (tenure / 112.0))
    out['dec112_rate'] = dec / normw.clip(lower=1e-6) * 7.0
    return out

newf = agent_api.build_features(fn).reset_index()
print("new feats:", newf.shape, newf.columns.tolist())

t = agent_api.load_saved('e013_stationary.parquet')
drops = ['week_sin','week_cos','trend','redemp_84','homeowner_code','age_code','class3_code','has_demo',
         'z_dec_224','spend_364','trips_364','dec_112','z_zero_block_share','r_84_364']
t2 = t.drop(columns=drops).merge(newf, on=['household_key','snapshot_day'], how='inner')
print("merged:", t2.shape, "dups:", t2[['household_key','snapshot_day']].duplicated().sum())

tt = agent_api.train_targets()
tr = t2.merge(tt, on=['household_key','snapshot_day']).query('snapshot_day <= 431')
for c in ['rate_364','rate_182','rate_112','dec112_rate','trips_rate_364','tenure_cap364']:
    ok = tr[c].notna()
    print(c, "corr:", round(np.corrcoef(tr.loc[ok,c], tr.loc[ok,'future_spend_4w'])[0,1],3))

g = t2.groupby('snapshot_day')[['rate_364','dec112_rate','tenure_cap364']].mean().round(3)
print(g.T)

path = agent_api.save_table(t2, 'e015_norm_windows')
print(path)


# ---- cell ----
import agent_api, pandas as pd
t2 = agent_api.load_saved('e015_norm_windows.parquet').drop(columns=['index'])
print(t2.shape, t2.columns.tolist()[-8:])
path = agent_api.save_table(t2, 'e015_norm_windows')
print(path)
