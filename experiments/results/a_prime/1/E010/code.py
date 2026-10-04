import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
print('e3 shape', e3.shape)
print('e3 cols:', list(e3.columns))

tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
print('merged', df.shape, 'missing target', int(df['future_spend_4w'].isna().sum()))
y = df['future_spend_4w']
print(y.describe(percentiles=[.1,.25,.5,.75,.9,.95,.99]))
print('zero share', float((y==0).mean()))

g = df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count'])
print(g)
print('std of snapshot means:', round(float(g['mean'].std()),2), 'overall mean:', round(float(y.mean()),2))

spend_cols = [c for c in e3.columns if 'spend' in c.lower()]
print('spend cols:', spend_cols)
def mae(p): return float((df[p]-y).abs().mean())
for c in spend_cols:
    print(c, 'MAE', round(mae(c),3), 'corr', round(float(df[c].corr(y)),3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

# 1) spend_l13 validity: NaN rate by snapshot, corr among tenure>=364
print('spend_l13 NaN rate by snapshot:')
print(tr.groupby('snapshot_day').spend_l13.apply(lambda s: round(float(s.isna().mean()),3)))
sub = tr[tr.spend_l13.notna()]
print('n valid l13:', len(sub), 'corr l13 vs y:', round(float(sub.spend_l13.corr(sub.future_spend_4w)),3),
      'MAE l13 alone:', round(float((sub.spend_l13-sub.future_spend_4w).abs().mean()),2))
sub2 = sub[sub.tenure>=364]
print('tenure>=364 n:', len(sub2), 'corr:', round(float(sub2.spend_l13.corr(sub2.future_spend_4w)),3))

# 2) simple ridge on core features -> residuals; then corr of candidates with residual
feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
X = tr[feats].fillna(0).values
X = np.column_stack([np.ones(len(X)), X])
# standardize
mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
lam = 10.0
A_ = Xs.T@Xs + lam*np.eye(Xs.shape[1]); A_[0,0]-=lam
w = np.linalg.solve(A_, Xs.T@y)
pred = Xs@w
res = y - pred
print('ridge train MAE:', round(float(np.abs(res).mean()),2))

# candidates to correlate with |res| and res
cands = ['spend_l13','spend_total','disc_share84','evening_share84','stores_l1','prods_l1','prods_l2',
         'avg_basket_l2','avg_basket_l3','days_active_l2','days_active_l3','active_share_l1','spend_l456_mean',
         'spend28_GROCERY','share84_GROCERY','share84_PRODUCE','share84_MEAT','private_share84','div84']
for c in cands:
    v = tr[c].fillna(0).values
    print(c, 'corr_res', round(float(np.corrcoef(v,res)[0,1]),3), 'corr_absres', round(float(np.corrcoef(v,np.abs(res))[0,1]),3))
print('corr pred vs y:', round(float(np.corrcoef(pred,y)[0,1]),3))
print('res std:', round(float(res.std()),2), 'y std:', round(float(y.std()),2))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
print(e3.groupby('snapshot_day').tenure.describe()[['min','25%','50%','75%','max']])

# how was spend_l13 filled when tenure short?
sub = e3[e3.tenure < 364]
print('tenure<364 rows:', len(sub), 'spend_l13 stats:', sub.spend_l13.describe()[['min','mean','max']].to_dict())
print('spend_l13 == 0 share among tenure<364:', float((sub.spend_l13==0).mean()))
print('spend_l13 == spend_l1 share:', float(np.isclose(sub.spend_l13, sub.spend_l1).mean()))
print('spend_l13 == spend_l123_mean share:', float(np.isclose(sub.spend_l13, sub.spend_l123_mean).mean()))
# among tenure>=364, is l13 nonzero?
sub2 = e3[e3.tenure >= 364]
print('tenure>=364 rows:', len(sub2), 'l13 zero share:', float((sub2.spend_l13==0).mean()))
print(e3.groupby('snapshot_day').apply(lambda g: pd.Series({'n':len(g), 'tenure>=364': float((g.tenure>=364).mean())}), include_groups=False))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

# 1) simple per-snapshot ridge on E003 core features: does snapshot-level recalibration help?
feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
# leave-one-snapshot-out CV within train for per-snapshot calibration
snaps = sorted(tr.snapshot_day.unique())
def fit_pred(train_mask, Xall, yall):
    X = Xall[train_mask]; yy = yall[train_mask]
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    lam=10.0
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    w = np.linalg.solve(Am, Xs.T@yy)
    Xt = Xall[~train_mask]
    Xts = np.column_stack([np.ones(len(Xt)), (Xt[:,1:]-mu)/sd])
    return Xts@w, Xs@w

Xall = tr[feats].fillna(0).values.astype(float)
yall = y
# global fit
pred_all, _ = fit_pred(np.ones(len(tr),bool), Xall, yall)
print('global ridge train MAE:', round(float(np.abs(pred_all-yall).mean()),2))

# per-snapshot intercept-only recalibration via LOSO
resid_by_snap = {}
for s in snaps:
    m = tr.snapshot_day==s
    p_in, p_all = None, None
    # fit on all except s
    mask = ~m.values
    w_pred, w_in = fit_pred(mask, Xall, yall)
    resid_by_snap[s] = float((yall[m.values]-w_pred).mean())
print('per-snapshot mean residual (LOSO):')
print({k: round(v,1) for k,v in resid_by_snap.items()})
print('spread:', round(max(resid_by_snap.values())-min(resid_by_snap.values()),1))

# 2) does adding snapshot_day (linear) change anything? quick check: corr of resid with snapshot
res = yall - pred_all
print('corr resid vs snapshot_day:', round(float(np.corrcoef(tr.snapshot_day, res)[0,1]),3))

# 3) winsorize y at 99th pct of train -> effect on achievable MAE (sanity, not usable directly)
cap = np.percentile(y, 99)
print('y capped at 99th:', round(float(np.abs(np.minimum(pred_all, cap)-yall).mean()),2))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
Xall = tr[feats].fillna(0).values.astype(float)

def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    w = np.linalg.solve(Am, Xs.T@yy)
    return w, mu, sd

def apply_w(w, mu, sd, X):
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    return Xs@w

w, mu, sd = fit_w(Xall, y)
pred_all = apply_w(w, mu, sd, Xall)
print('global ridge train MAE:', round(float(np.abs(pred_all-y).mean()),2))

snaps = sorted(tr.snapshot_day.unique())
resid_by_snap = {}
for s in snaps:
    m = (tr.snapshot_day==s).values
    w2, mu2, sd2 = fit_w(Xall[~m], y[~m])
    p = apply_w(w2, mu2, sd2, Xall[m])
    resid_by_snap[s] = float((y[m]-p).mean())
print('per-snapshot mean residual (LOSO):', {k: round(v,1) for k,v in resid_by_snap.items()})
print('spread:', round(max(resid_by_snap.values())-min(resid_by_snap.values()),1))

res = y - pred_all
print('corr resid vs snapshot_day:', round(float(np.corrcoef(tr.snapshot_day, res)[0,1]),3))
cap = np.percentile(y, 99)
print('y capped at 99th pct -> MAE:', round(float(np.abs(np.minimum(pred_all, cap)-y).mean()),2))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
Xall = tr[feats].fillna(0).values.astype(float)
def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w
w, mu, sd = fit_w(Xall, y)
pred = apply_w(w, mu, sd, Xall)

zero = y==0
print('y=0 rows:', zero.sum(), 'mean pred on them:', round(float(pred[zero].mean()),1),
      'MAE contribution:', round(float(np.abs(pred[zero]).mean()*zero.mean()),2))
print('y>0 rows: mean |err|:', round(float(np.abs((pred-y))[~zero]).mean(),1) if False else round(float(np.abs(pred[~zero]-y[~zero]).mean()),2))

# how separable are y=0 households?
print('\nAmong y=0: spend_l1 quantiles:', np.percentile(tr.spend_l1[zero], [10,25,50,75,90]).round(1))
print('Among y>0: spend_l1 quantiles:', np.percentile(tr.spend_l1[~zero], [10,25,50,75,90]).round(1))
print('spend_l1==0 -> P(y=0):', round(float((y[tr.spend_l1==0]==0).mean()),3), 'n=', int((tr.spend_l1==0).sum()))
print('spend_l1>0 -> P(y=0):', round(float((y[tr.spend_l1>0]==0).mean()),3))
print('days_since_last>28 -> P(y=0):', round(float((y[tr.days_since_last>28]==0).mean()),3), 'n=', int((tr.days_since_last>28).sum()))
print('days_since_last<=28 -> P(y=0):', round(float((y[tr.days_since_last<=28]==0).mean()),3))

# error decomposition: how much would perfect zero-classification help?
# if we could perfectly identify y=0 and set pred=0 there:
pred2 = pred.copy(); pred2[zero]=0
print('\nMAE now:', round(float(np.abs(pred-y).mean()),2), 'MAE with oracle zero:', round(float(np.abs(pred2-y).mean()),2))
# if we just clipped predictions at 0..cap: no change. What about shrink of positives?
# contribution of y=0 rows to MAE:
print('MAE on y=0 rows:', round(float(np.abs(pred[zero]).mean()),2), 'weight', round(float(zero.mean()),3))
print('MAE on y>0 rows:', round(float(np.abs(pred[~zero]-y[~zero]).mean()),2))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

def make_micro(view, sd):
    hh = view.households
    if hasattr(hh, 'columns'):
        keys = pd.Index(hh['household_key'].unique()) if 'household_key' in hh.columns else pd.Index(hh.index)
    else:
        keys = pd.Index(hh)
    idx = keys
    tx = view.transactions
    tx = tx[tx.day > sd - 380]
    out = pd.DataFrame(index=idx)

    def agg(mask, col, how):
        t = tx[mask]
        s = t.groupby('household_key')[col].sum() if how=='sum' else t.groupby('household_key')[col].nunique()
        return s.reindex(idx).fillna(0.0)

    spends, trips = {}, {}
    for k in range(1,13):
        hi = sd - 28*(k-1); lo = sd - 28*k
        m = (tx.day>lo)&(tx.day<=hi)
        spends[k] = agg(m,'sales_value','sum')
        if k<=6: trips[k] = agg(m,'basket_id','nunique')
    for k in range(1,7):
        out[f'mspend_l{k}'] = spends[k]; out[f'mtrips_l{k}'] = trips[k]
    out['spend_7d']  = agg((tx.day>sd-7)&(tx.day<=sd),'sales_value','sum')
    out['spend_14d'] = agg((tx.day>sd-14)&(tx.day<=sd),'sales_value','sum')
    out['trips_7d']  = agg((tx.day>sd-7)&(tx.day<=sd),'basket_id','nunique')
    out['trips_14d'] = agg((tx.day>sd-14)&(tx.day<=sd),'basket_id','nunique')
    out['n_zero_w6']  = sum((spends[k]==0).astype(float) for k in range(1,7))
    out['n_zero_w12'] = sum((spends[k]==0).astype(float) for k in range(1,13))
    S = pd.concat([spends[k] for k in range(1,7)], axis=1)
    mu = S.mean(1); med = S.median(1)
    out['spend_cv_l6'] = (S.std(1)/(mu+1e-6)).where(mu>0, 0.0)
    out['spend_max_over_med_l6'] = (S.max(1)/(med+1e-6)).where(med>0, 3.0)
    # inter-trip gaps, last 168d
    t2 = tx[['household_key','day']].drop_duplicates().sort_values(['household_key','day'])
    t2['gap'] = t2.groupby('household_key').day.diff()
    rec = t2[t2.day > sd-168]
    gs = rec.groupby('household_key').gap.agg(['mean','std','max'])
    out['gap_mean_l6'] = gs['mean'].reindex(idx)
    out['gap_std_l6']  = gs['std'].reindex(idx)
    out['gap_max_l6']  = gs['max'].reindex(idx)
    out['n_gap21_l6']  = (rec.gap>=21).groupby(rec.household_key).sum().reindex(idx).fillna(0.0)
    out['dsl'] = sd - tx.groupby('household_key').day.max().reindex(idx)
    # 84d basket micro-structure
    t84 = tx[tx.day > sd-84]
    b84 = t84.groupby(['household_key','basket_id']).sales_value.sum().rename('v').reset_index()
    bday = t84.groupby('basket_id').day.max()
    bmed = b84.groupby('household_key').v.median(); bmean = b84.groupby('household_key').v.mean()
    bmax = b84.groupby('household_key').v.max(); bstd = b84.groupby('household_key').v.std()
    out['basket_med_84'] = bmed.reindex(idx).fillna(0.0)
    out['basket_max_84'] = bmax.reindex(idx).fillna(0.0)
    out['basket_cv_84'] = (bstd/(bmean+1e-6)).reindex(idx).fillna(0.0)
    out['basket_max_over_med_84'] = (bmax/(bmed+1e-6)).reindex(idx)
    bb = b84.copy(); bb['day'] = bb.basket_id.map(bday); bb['med'] = bb.household_key.map(bmed)
    big = bb[bb.v > 2*bb.med]
    out['n_stockup_84'] = big.groupby('household_key').size().reindex(idx).fillna(0.0)
    out['dsl_stockup'] = (sd - big.groupby('household_key').day.max().reindex(idx)).fillna(999.0)
    sp84 = t84.groupby('household_key').sales_value.sum().reindex(idx).fillna(0.0)
    out['stockup_share_84'] = big.groupby('household_key').v.sum().reindex(idx).fillna(0.0)/(sp84+1e-6)
    trips84 = t84.groupby('household_key').basket_id.nunique().reindex(idx).fillna(0.0)
    units84 = t84.groupby('household_key').quantity.sum().reindex(idx).fillna(0.0)
    out['units_84'] = units84
    out['unit_price_84'] = (sp84/(units84.abs()+1e-6)).clip(-50,200)
    out['units_per_trip_84'] = units84.abs()/(trips84+1e-6)
    t84b = t84[['household_key','basket_id','day']].drop_duplicates()
    t84b = t84b.assign(we=(t84b.day%7).isin([5,6]))
    out['weekend_share_84'] = t84b.groupby('household_key').we.mean().reindex(idx)
    out['stores_84'] = t84.groupby('household_key').store_id.nunique().reindex(idx).fillna(0.0)
    out['morning_share_84'] = (t84.trans_time<1200).groupby(t84.household_key).mean().reindex(idx)
    return out.astype(float)

micro = A.build_features(make_micro)
print(micro.shape)
print(micro.head(3).T)
A.save_table(micro, 'micro.parquet')
print('saved')


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

micro = A.load_saved('micro.parquet')
tt = A.train_targets()
df = micro.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

cols = [c for c in micro.columns if c not in ('household_key','snapshot_day')]
# univariate: corr with y, MAE of column alone (as predictor), corr with |resid| of a simple base
base_feats = ['mspend_l1','mspend_l2','mspend_l3','mspend_l4','mspend_l5','mspend_l6','mtrips_l1','mtrips_l2','mtrips_l3',
              'spend_7d','spend_14d','dsl','n_zero_w6','n_zero_w12','tenure' if 'tenure' in tr else 'mspend_l6']
def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w

bf = [c for c in ['mspend_l1','mspend_l2','mspend_l3','mspend_l4','mspend_l5','mspend_l6','mtrips_l1','mtrips_l2','mtrips_l3'] if c in tr]
Xb = tr[bf].fillna(0).values.astype(float)
w, mu, sd = fit_w(Xb, y)
res = y - apply_w(w, mu, sd, Xb)
print('base train MAE:', round(float(np.abs(res).mean()),2))

rows=[]
for c in cols:
    v = tr[c].fillna(0).values if tr[c].dtype.kind in 'fc' else tr[c].fillna('NA')
    if tr[c].dtype.kind in 'fc':
        corr = float(np.corrcoef(v, y)[0,1]) if np.std(v)>0 else 0.0
        corr_res = float(np.corrcoef(v, res)[0,1]) if np.std(v)>0 else 0.0
        corr_abs = float(np.corrcoef(v, np.abs(res))[0,1]) if np.std(v)>0 else 0.0
        # alone-MAE
        w1, mu1, sd1 = fit_w(v.reshape(-1,1), y)
        p1 = apply_w(w1, mu1, sd1, v.reshape(-1,1))
        m1 = float(np.abs(p1-y).mean())
        rows.append((c, round(corr,3), round(corr_res,3), round(corr_abs,3), round(m1,2)))
r = pd.DataFrame(rows, columns=['feat','corr_y','corr_res','corr_absres','alone_mae']).sort_values('alone_mae')
print(r.to_string(index=False))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e003_catmix.parquet')
e3 = e3.copy()
fake = e3.tenure < 364
print('rows with fake l13:', int(fake.sum()), 'of', len(e3))
e3['spend_l13'] = e3.spend_l13.where(~fake, np.nan)
e3['has_real_l13'] = (~fake).astype(float)
# also a ratio of seasonal lag to recent mean (only meaningful when real)
e3['l13_over_recent'] = e3.spend_l13 / (e3.spend_l123_mean + 1e-6)
print(e3[['spend_l13','has_real_l13','l13_over_recent']].describe())
A.save_table(e3, 'e010_l13fix.parquet')
print('saved', e3.shape)


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e010_l13fix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
# only rows with REAL l13
sub = tr[tr.has_real_l13==1]
print('n real-l13 train rows:', len(sub))
print('corr l13 vs y:', round(float(sub.spend_l13.corr(sub.future_spend_4w)),3))
w = np.polyfit(sub.spend_l13, sub.future_spend_4w, 1)
print('linear fit: y =', round(w[0],3), '* l13 +', round(w[1],2))
p = np.polyval(w, sub.spend_l13)
print('alone MAE on these rows:', round(float(np.abs(p-sub.future_spend_4w).mean()),2))
print('MAE of spend_l123_mean on same rows:', round(float(np.abs(sub.spend_l123_mean-sub.future_spend_4w).mean()),2))
# ratio feature distribution (clip the crazy tail)
r = sub.l13_over_recent.clip(0,5)
print('l13_over_recent (clipped 0-5) quantiles:', np.percentile(r,[10,25,50,75,90]).round(2))
print('corr ratio vs y:', round(float(r.corr(sub.future_spend_4w)),3))
print('corr ratio vs spend_l123_mean:', round(float(r.corr(sub.spend_l123_mean)),3))
print('corr l13 vs l123_mean:', round(float(sub.spend_l13.corr(sub.spend_l123_mean)),3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e010_l13fix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
y = tr['future_spend_4w'].values

feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
Xall = tr[feats].fillna(0).values.astype(float)
def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w
w, mu, sd = fit_w(Xall, y)
pred = apply_w(w, mu, sd, Xall)
res = y - pred

# residual corr among REAL l13 rows
m = tr.has_real_l13==1
print('corr resid vs l13 (real rows):', round(float(np.corrcoef(tr.spend_l13[m], res[m.values])[0,1]),3))
print('corr resid vs l13_over_recent (real):', round(float(np.corrcoef(tr.l13_over_recent[m].clip(-5,5), res[m.values])[0,1]),3))
print('corr resid vs has_real_l13:', round(float(np.corrcoef(tr.has_real_l13, res)[0,1]),3))

# what fraction of val rows would be affected?
val = df[df.future_spend_4w.isna()]
print('val rows with real l13:', int((val.has_real_l13==1).sum()), '/', len(val))
# distribution shift check: spend_l123_mean train vs val
print('train l123 mean:', round(float(tr.spend_l123_mean.mean()),1), 'val:', round(float(val.spend_l123_mean.mean()),1))
print('train tenure mean:', round(float(tr.tenure.mean()),1), 'val:', round(float(val.tenure.mean()),1))
# key: among val rows, how many have tenure < 364 (i.e., would have had fake l13 in E003)?
print('val rows tenure<364:', int((val.tenure<364).sum()), 'of', len(val))
# check l13 vs y relationship stability across snapshots among real rows
for s in sorted(sub.snapshot_day.unique()):
    ss = sub[sub.snapshot_day==s]
    print(s, 'n', len(ss), 'corr', round(float(ss.spend_l13.corr(ss.future_spend_4w)),3), 'mean y', round(float(ss.future_spend_4w.mean()),1))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

e3 = A.load_saved('e010_l13fix.parquet')
tt = A.train_targets()
df = e3.merge(tt, on=['household_key','snapshot_day'], how='left')
tr = df[df.future_spend_4w.notna()].copy()
sub = tr[tr.has_real_l13==1]
for s in sorted(sub.snapshot_day.unique()):
    ss = sub[sub.snapshot_day==s]
    print(s, 'n', len(ss), 'corr', round(float(ss.spend_l13.corr(ss.future_spend_4w)),3), 'mean y', round(float(ss.future_spend_4w.mean()),1))
# l13 vs l123_mean among real rows: is l13 just a noisy duplicate?
print('\nreal rows: corr l13 vs l123_mean:', round(float(sub.spend_l13.corr(sub.spend_l123_mean)),3))
# per-household: does l13 add info beyond l1..l6? quick ridge with/without
feats = ['spend_l1','spend_l2','spend_l3','spend_l4','spend_l5','spend_l6','trips_l1','trips_l2','trips_l3',
         'avg_basket_l1','days_active_l1','days_since_last','tenure','spend_rate28','momentum','zero_recent',
         'trend_1v2','trend_1v3','div84','max_share84','active_share_l1']
def fit_w(X, yy, lam=10.0):
    mu, sd = X[:,1:].mean(0), X[:,1:].std(0)+1e-9
    Xs = np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])
    Am = Xs.T@Xs + lam*np.eye(Xs.shape[1]); Am[0,0]-=lam
    return np.linalg.solve(Am, Xs.T@yy), mu, sd
def apply_w(w, mu, sd, X):
    return np.column_stack([np.ones(len(X)), (X[:,1:]-mu)/sd])@w
Xa = sub[feats].fillna(0).values.astype(float); ya = sub.future_spend_4w.values
w0,mu0,sd0 = fit_w(Xa, ya); p0 = apply_w(w0,mu0,sd0,Xa)
Xb = np.column_stack([Xa, sub.spend_l13.fillna(0).values]); 
w1,mu1,sd1 = fit_w(Xb, ya); p1 = apply_w(w1,mu1,sd1,Xb)
print('train MAE without l13:', round(float(np.abs(p0-ya).mean()),2), 'with l13:', round(float(np.abs(p1-ya).mean()),2))
# LOSO-style: hold out last snapshot among real rows
smax = sub.snapshot_day.max()
m = sub.snapshot_day==smax
w2,mu2,sd2 = fit_w(Xa[~m.values], ya[~m.values]); p2 = apply_w(w2,mu2,sd2,Xa[m.values])
w3,mu3,sd3 = fit_w(np.column_stack([Xa, sub.spend_l13.fillna(0).values])[~m.values], ya[~m.values]); p3 = apply_w(w3,mu3,sd3,np.column_stack([Xa, sub.spend_l13.fillna(0).values])[m.values])
print('holdout(431) MAE without l13:', round(float(np.abs(p2-ya[m.values]).mean()),2), 'with l13:', round(float(np.abs(p3-ya[m.values]).mean()),2))
