
import agent_api as A, numpy as np, pandas as pd

oof = A.load_saved('oof_e013.parquet')
print('oof shape', oof.shape, 'cols', oof.columns.tolist())
print(oof.head(3))
print(oof.groupby('snapshot_day').size())

tt = A.train_targets()
print('targets', tt.shape)
m = tt.merge(oof, on=['household_key','snapshot_day'], how='inner')
print('merged', m.shape)
pcol = [c for c in m.columns if c not in ('household_key','snapshot_day','future_spend_4w')][0]
print('pred col', pcol)
print('OOF MAE e013:', np.abs(m[pcol]-m.future_spend_4w).mean())
g = m.groupby('snapshot_day').apply(lambda d: pd.Series({'tgt_mean':d.future_spend_4w.mean(),'pred_mean':d[pcol].mean(),'bias':(d[pcol]-d.future_spend_4w).mean(),'mae':np.abs(d[pcol]-d.future_spend_4w).mean()}))
print(g)

feats = A.load_saved('feats_v4.parquet')
print('feats shape', feats.shape)
print('feats cols', feats.columns.tolist())


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
tr = feats.merge(tt, on=['household_key','snapshot_day'])
print('train rows', len(tr))
y = tr.future_spend_4w.values
print('y: mean %.1f median %.1f p90 %.1f p99 %.1f zero-share %.3f' % (y.mean(), np.median(y), np.percentile(y,90), np.percentile(y,99), (y==0).mean()))

# naive predictors
def mae(p): return np.abs(np.asarray(p)-y).mean()
print('const median %.2f | const mean %.2f | zero %.2f' % (mae(np.median(y)), mae(y.mean()), mae(np.zeros_like(y))))
for c in ['exp4w_blend','lag1_spend','spend_28','exp4w_all','exp4w_84']:
    print(c, 'MAE %.2f' % mae(tr[c].values))
oof = A.load_saved('oof_e013.parquet')
tr2 = tr.merge(oof[['household_key','snapshot_day','oof']], on=['household_key','snapshot_day'])
print('oof MAE %.3f' % mae(tr2.oof.values))
for w in [0.1,0.2,0.3]:
    print('oof*(1-w)+exp4w_blend*w, w=%.1f MAE %.3f' % (w, mae((1-w)*tr2.oof.values + w*tr2.exp4w_blend.values)))
# residual stats
res = y - tr.exp4w_blend.values
print('residual: mean %.1f median %.1f std %.1f' % (res.mean(), np.median(res), res.std()))
print('corr(oof, y) %.3f | corr(oof, exp4w_blend) %.3f' % (np.corrcoef(tr2.oof, tr2.future_spend_4w)[0,1], np.corrcoef(tr2.oof, tr2.exp4w_blend)[0,1]))

# saved val preds: correlations
ps = {n: A.load_saved('pred_e%03d.parquet'%n) for n in [4,5,7,11,13,19]}
base = ps[13][['household_key','snapshot_day']].copy()
m = base.copy()
for n,p in ps.items():
    m = m.merge(p.rename(columns={'prediction':'p%d'%n}), on=['household_key','snapshot_day'])
print(m.shape)
print(m[[c for c in m.columns if c.startswith('p')]].corr().round(3))
for n in [4,5,7,11,19]:
    print('blend e013+e%03d 50/50 MAE:'%n, end=' ')
    mm = m.merge(tt, on=['household_key','snapshot_day']) if False else m
    # no targets for val; just corr shown above
print()
import time
t0=time.time()
import xgboost as xgb
X = tr[[c for c in feats.columns if c not in ('household_key','snapshot_day')]].head(20000)
dtr = xgb.DMatrix(X, label=y[:20000])
b = xgb.train({'objective':'reg:quantileerror','quantile_alpha':0.5,'max_depth':5,'min_child_weight':40,'learning_rate':0.08,'nthread':4}, dtr, 400)
print('one xgb fit (20k rows, 400 trees): %.1fs' % (time.time()-t0))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd

oof = A.load_saved('oof_e013.parquet')
p = oof.oof.values; y = oof.y.values
print('OOF MAE base: %.4f' % np.abs(p-y).mean())

def calib_mae(p, y, nbins, lam, folds=5, seed=0):
    # piecewise conditional-median calibration + shrinkage toward identity, honest via CV
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(p)); f = np.array_split(idx, folds)
    preds = np.empty(len(p))
    qs = np.quantile(p, np.linspace(0,1,nbins+1))
    for te in f:
        tr = np.setdiff1d(idx, te, assume_unique=False)
        pt, yt = p[tr], y[tr]
        bins = np.clip(np.searchsorted(qs[1:-1], pt, side='right'), 0, nbins-1)
        med = np.array([np.median(yt[bins==b]) if (bins==b).sum()>30 else np.nan for b in range(nbins)])
        # fill empty bins by interpolation from bin centers
        bc = np.array([(qs[b]+qs[b+1])/2 for b in range(nbins)])
        ok = ~np.isnan(med)
        med = np.interp(bc, bc[ok], med[ok])
        bt = np.clip(np.searchsorted(qs[1:-1], p[te], side='right'), 0, nbins-1)
        g = med[bt]
        preds[te] = lam*p[te] + (1-lam)*g
    return np.abs(preds-y).mean(), preds

best = None
for nbins in [6,10,15,25]:
    for lam in [0.0,0.2,0.4,0.6,0.8]:
        m,_ = calib_mae(p,y,nbins,lam)
        if best is None or m < best[0]: best = (m,nbins,lam)
        print('nbins %2d lam %.1f -> CV MAE %.4f' % (nbins,lam,m))
print('BEST:', best)

# global shift only, for reference
mshift = np.median(y-p)
print('shift-only (add median residual): CV-free MAE %.4f' % np.abs(p+mshift-y).mean())
# 2D check: does adding spend_28 to calibration help?
feats = A.load_saved('feats_v4.parquet')
o2 = oof.merge(feats[['household_key','snapshot_day','spend_28','exp4w_blend']], on=['household_key','snapshot_day'])
print(o2.shape)


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')[['household_key','day','product_id','quantity','sales_value','store_id','week_no']].copy()
print('tx rows', len(tx))
dm = v.table('display_mailer')
print('dm shape', dm.shape)
print('display vc', dm.display.value_counts(dropna=False).head(8).to_dict())
print('mailer vc', dm.mailer.value_counts(dropna=False).head(8).to_dict())

def bmask(s):
    sn = pd.to_numeric(s, errors='coerce')
    if sn.notna().all(): return (sn>0).values
    return (~s.astype(str).str.strip().isin(['0','0.0','','nan','None'])).values

dmk = dm[['product_id','store_id','week_no']].copy()
dmk['d'] = bmask(dm.display); dmk['m'] = bmask(dm.mailer)
dm_d = dmk[dmk.d][['product_id','store_id','week_no']].drop_duplicates(); dm_d['ed']=1.0
dm_m = dmk[dmk.m][['product_id','store_id','week_no']].drop_duplicates(); dm_m['em']=1.0
print('dm_d', dm_d.shape, 'dm_m', dm_m.shape)

prod = v.table('products')[['product_id','commodity_desc']]

feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
train_days = sorted(tt.snapshot_day.unique())
out = []
for d in train_days:
    w = tx[tx.day<=d]
    w84 = w[w.day>d-84]; w28 = w[w.day>d-28]; w7 = w[w.day>d-7]
    g = w84.groupby('household_key')
    f = pd.DataFrame(index=g.size().index)
    f['units_84'] = g.quantity.sum()
    f['spend84'] = g.sales_value.sum()
    f['trips_7'] = w7.groupby('household_key').basket_id.nunique()
    f['nprod_28'] = w28.groupby('household_key').product_id.nunique()
    w28c = w28.merge(prod, on='product_id', how='left')
    f['ncomm_28'] = w28c.groupby('household_key').commodity_desc.nunique()
    q = w84.quantity.where(w84.quantity>0)
    f['up_mean_84'] = (w84.sales_value/q).groupby(w84.household_key).mean()
    glob_up = (w84.sales_value/q).mean()
    f['premium_idx'] = f.up_mean_84/glob_up
    # store richness: spend per household at store (84d)
    st = w84.groupby('store_id').agg(sv=('sales_value','sum'), nh=('household_key','nunique'))
    st['rich'] = st.sv/st.nh
    ss = w84.groupby(['household_key','store_id'], as_index=False).sales_value.sum()
    top = ss.sort_values('sales_value').groupby('household_key').tail(1).set_index('household_key')
    f['store_rich_84'] = st.rich.reindex(top.store_id).values
    # exposure joins
    j = w84.merge(dm_d, on=['product_id','store_id','week_no'], how='left')
    j['ed'] = j.ed.fillna(0.0); j['sve'] = j.sales_value*j.ed
    den = j.groupby('household_key').sales_value.sum()
    f['disp_share_84'] = j.groupby('household_key').sve.sum()/den
    j2 = w84.merge(dm_m, on=['product_id','store_id','week_no'], how='left')
    j2['em'] = j2.em.fillna(0.0); j2['svm'] = j2.sales_value*j2.em
    f['mail_share_84'] = j2.groupby('household_key').svm.sum()/den
    f['snapshot_day'] = d
    out.append(f.reset_index().rename(columns={'index':'household_key'}))
cand = pd.concat(out, ignore_index=True)
print('cand', cand.shape, 'elapsed %.0fs' % (time.time()-t0))

oof = A.load_saved('oof_e013.parquet')
m = tt.merge(oof, on=['household_key','snapshot_day']).merge(cand, on=['household_key','snapshot_day'])
m['resid'] = m.future_spend_4w - m.oof
m['r_7_28'] = m.spend_7/(m.spend_28+1); m['r_28_84'] = m.spend_28/(m.spend_84+1); m['r_84_168'] = m.spend_84/(m.spend_168+1)
m['lag_std'] = m[['lag1_spend','lag2_spend','lag3_spend']].std(axis=1)
W = m[['wk%d'%i for i in range(8)]].values; t8 = np.arange(8)
m['wk_slope'] = ((W*t8).sum(1)*8 - W.sum(1)*t8.sum())/(8*(t8**2).sum()-t8.sum()**2)
cc = [c for c in m.columns if c not in ('household_key','snapshot_day','oof','y','future_spend_4w','resid')]
for c in cc:
    x = m[c].fillna(m[c].median())
    print('%-16s corr_resid %+.4f  corr_y %+.4f' % (c, np.corrcoef(x, m.resid)[0,1], np.corrcoef(x, m.future_spend_4w)[0,1]))
A.save_table(cand, 'cand_train.parquet')
print('saved. elapsed %.0fs' % (time.time()-t0))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')[['household_key','day','basket_id','product_id','quantity','sales_value','store_id','week_no']].copy()
dm = v.table('display_mailer')
def bmask(s):
    sn = pd.to_numeric(s, errors='coerce')
    if sn.notna().all(): return (sn>0).values
    return (~s.astype(str).str.strip().isin(['0','0.0','','nan','None'])).values
dmk = dm[['product_id','store_id','week_no']].copy()
dmk['d'] = bmask(dm.display); dmk['m'] = bmask(dm.mailer)
dm_d = dmk[dmk.d][['product_id','store_id','week_no']].drop_duplicates(); dm_d['ed']=1.0
dm_m = dmk[dmk.m][['product_id','store_id','week_no']].drop_duplicates(); dm_m['em']=1.0
prod = v.table('products')[['product_id','commodity_desc']]
feats = A.load_saved('feats_v4.parquet')
tt = A.train_targets()
train_days = sorted(tt.snapshot_day.unique())
out = []
for d in train_days:
    w = tx[tx.day<=d]
    w84 = w[w.day>d-84]; w28 = w[w.day>d-28]; w7 = w[w.day>d-7]
    g = w84.groupby('household_key')
    f = pd.DataFrame(index=g.size().index)
    f['units_84'] = g.quantity.sum()
    f['spend84'] = g.sales_value.sum()
    f['trips_7'] = w7.groupby('household_key').basket_id.nunique()
    f['nprod_28'] = w28.groupby('household_key').product_id.nunique()
    w28c = w28.merge(prod, on='product_id', how='left')
    f['ncomm_28'] = w28c.groupby('household_key').commodity_desc.nunique()
    q = w84.quantity.where(w84.quantity>0)
    f['up_mean_84'] = (w84.sales_value/q).groupby(w84.household_key).mean()
    glob_up = (w84.sales_value/q).mean()
    f['premium_idx'] = f.up_mean_84/glob_up
    st = w84.groupby('store_id').agg(sv=('sales_value','sum'), nh=('household_key','nunique'))
    st['rich'] = st.sv/st.nh
    ss = w84.groupby(['household_key','store_id'], as_index=False).sales_value.sum()
    top = ss.sort_values('sales_value').groupby('household_key').tail(1).set_index('household_key')
    f['store_rich_84'] = st.rich.reindex(top.store_id).values
    j = w84.merge(dm_d, on=['product_id','store_id','week_no'], how='left')
    j['ed'] = j.ed.fillna(0.0); j['sve'] = j.sales_value*j.ed
    den = j.groupby('household_key').sales_value.sum()
    f['disp_share_84'] = j.groupby('household_key').sve.sum()/den
    j2 = w84.merge(dm_m, on=['product_id','store_id','week_no'], how='left')
    j2['em'] = j2.em.fillna(0.0); j2['svm'] = j2.sales_value*j2.em
    f['mail_share_84'] = j2.groupby('household_key').svm.sum()/den
    f['snapshot_day'] = d
    out.append(f.reset_index().rename(columns={'index':'household_key'}))
cand = pd.concat(out, ignore_index=True)
print('cand', cand.shape, 'elapsed %.0fs' % (time.time()-t0))
oof = A.load_saved('oof_e013.parquet')
m = tt.merge(oof, on=['household_key','snapshot_day']).merge(cand, on=['household_key','snapshot_day'])
m['resid'] = m.future_spend_4w - m.oof
m['r_7_28'] = m.spend_7/(m.spend_28+1); m['r_28_84'] = m.spend_28/(m.spend_84+1); m['r_84_168'] = m.spend_84/(m.spend_168+1)
m['lag_std'] = m[['lag1_spend','lag2_spend','lag3_spend']].std(axis=1)
W = m[['wk%d'%i for i in range(8)]].values; t8 = np.arange(8)
m['wk_slope'] = ((W*t8).sum(1)*8 - W.sum(1)*t8.sum())/(8*(t8**2).sum()-t8.sum()**2)
cc = [c for c in m.columns if c not in ('household_key','snapshot_day','oof','y','future_spend_4w','resid')]
for c in cc:
    x = m[c].fillna(m[c].median())
    print('%-16s corr_resid %+.4f  corr_y %+.4f' % (c, np.corrcoef(x, m.resid)[0,1], np.corrcoef(x, m.future_spend_4w)[0,1]))
A.save_table(cand, 'cand_train.parquet')
print('saved. elapsed %.0fs' % (time.time()-t0))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); oof = A.load_saved('oof_e013.parquet')
cand = A.load_saved('cand_train.parquet')
m = tt.merge(oof, on=['household_key','snapshot_day']).merge(feats, on=['household_key','snapshot_day']).merge(cand, on=['household_key','snapshot_day'])
print('merged', m.shape)
m['resid'] = m.future_spend_4w - m.oof
m['r_7_28'] = m.spend_7/(m.spend_28+1); m['r_28_84'] = m.spend_28/(m.spend_84+1); m['r_84_168'] = m.spend_84/(m.spend_168+1)
m['lag_std'] = m[['lag1_spend','lag2_spend','lag3_spend']].std(axis=1)
W = m[['wk%d'%i for i in range(8)]].values; t8 = np.arange(8)
m['wk_slope'] = ((W*t8).sum(1)*8 - W.sum(1)*t8.sum())/(8*(t8**2).sum()-t8.sum()**2)
base = ['spend_7','spend_28','spend_84','spend_168','lag1_spend','lag2_spend','lag3_spend','exp4w_blend','trips_28','trips_84','tenure','days_since_last','wk_mean8','wk_cv8','active_weeks8','snap_day','snap_cycle_pos']
cands = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84','r_7_28','r_28_84','r_84_168','lag_std','wk_slope']
for c in cands+base:
    x = m[c].fillna(m[c].median())
    print('%-16s corr_resid %+.4f  corr_y %+.4f  corr_oof %+.3f' % (c, np.corrcoef(x, m.resid)[0,1], np.corrcoef(x, m.future_spend_4w)[0,1], np.corrcoef(x, m.oof)[0,1]))
print(m[['disp_share_84','mail_share_84']].describe().round(4))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd, time
t0=time.time()
v = A.snapshot(459)
tx = v.table('transactions')[['household_key','day','basket_id','product_id','quantity','sales_value','store_id','week_no']].copy()
dm = v.table('display_mailer')
def bmask(s):
    sn = pd.to_numeric(s, errors='coerce')
    if sn.notna().all(): return (sn>0).values
    return (~s.astype(str).str.strip().isin(['0','0.0','','nan','None'])).values
dmk = dm[['product_id','store_id','week_no']].copy()
dmk['d'] = bmask(dm.display); dmk['m'] = bmask(dm.mailer)
dm_d = dmk[dmk.d][['product_id','store_id','week_no']].drop_duplicates(); dm_d['ed']=1.0
dm_m = dmk[dmk.m][['product_id','store_id','week_no']].drop_duplicates(); dm_m['em']=1.0
prod = v.table('products')[['product_id','commodity_desc']]
tt = A.train_targets()
train_days = sorted(tt.snapshot_day.unique())
out = []
for d in train_days:
    w = tx[tx.day<=d]
    w84 = w[w.day>d-84]; w28 = w[w.day>d-28]; w7 = w[w.day>d-7]
    g = w84.groupby('household_key')
    f = pd.DataFrame(index=g.size().index)
    f['units_84'] = g.quantity.sum()
    f['spend84'] = g.sales_value.sum()
    f['trips_7'] = w7.groupby('household_key').basket_id.nunique()
    f['nprod_28'] = w28.groupby('household_key').product_id.nunique()
    w28c = w28.merge(prod, on='product_id', how='left')
    f['ncomm_28'] = w28c.groupby('household_key').commodity_desc.nunique()
    q = w84.quantity.where(w84.quantity>0)
    f['up_mean_84'] = (w84.sales_value/q).groupby(w84.household_key).mean()
    glob_up = (w84.sales_value/q).mean()
    f['premium_idx'] = f.up_mean_84/glob_up
    st = w84.groupby('store_id').agg(sv=('sales_value','sum'), nh=('household_key','nunique'))
    st['rich'] = st.sv/st.nh
    ss = w84.groupby(['household_key','store_id'], as_index=False).sales_value.sum()
    top = ss.sort_values('sales_value').groupby('household_key').tail(1).set_index('household_key')
    f['store_rich_84'] = st.rich.reindex(top.store_id).values
    j = w84.merge(dm_d, on=['product_id','store_id','week_no'], how='left')
    j['ed'] = j.ed.fillna(0.0); j['sve'] = j.sales_value*j.ed
    den = j.groupby('household_key').sales_value.sum()
    f['disp_share_84'] = j.groupby('household_key').sve.sum()/den
    j2 = w84.merge(dm_m, on=['product_id','store_id','week_no'], how='left')
    j2['em'] = j2.em.fillna(0.0); j2['svm'] = j2.sales_value*j2.em
    f['mail_share_84'] = j2.groupby('household_key').svm.sum()/den
    f['snapshot_day'] = d
    out.append(f.reset_index().rename(columns={'index':'household_key'}))
cand = pd.concat(out, ignore_index=True)
A.save_table(cand, 'cand_train.parquet')
print('cand', cand.shape, 'saved OK. elapsed %.0fs' % (time.time()-t0))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); oof = A.load_saved('oof_e013.parquet')
cand = A.load_saved('cand_train.parquet')
m = tt.merge(oof, on=['household_key','snapshot_day']).merge(feats, on=['household_key','snapshot_day']).merge(cand, on=['household_key','snapshot_day'])
print('merged', m.shape)
m['resid'] = m.future_spend_4w - m.oof
m['r_7_28'] = m.spend_7/(m.spend_28+1); m['r_28_84'] = m.spend_28/(m.spend_84+1); m['r_84_168'] = m.spend_84/(m.spend_168+1)
m['lag_std'] = m[['lag1_spend','lag2_spend','lag3_spend']].std(axis=1)
W = m[['wk%d'%i for i in range(8)]].values; t8 = np.arange(8)
m['wk_slope'] = ((W*t8).sum(1)*8 - W.sum(1)*t8.sum())/(8*(t8**2).sum()-t8.sum()**2)
base = ['spend_7','spend_28','spend_84','spend_168','lag1_spend','lag2_spend','lag3_spend','exp4w_blend','trips_28','trips_84','tenure','days_since_last','wk_mean8','wk_cv8','active_weeks8','snap_day','snap_cycle_pos']
cands = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84','r_7_28','r_28_84','r_84_168','lag_std','wk_slope']
for c in cands+base:
    x = m[c].fillna(m[c].median())
    print('%-16s corr_resid %+.4f  corr_y %+.4f  corr_oof %+.3f' % (c, np.corrcoef(x, m.resid)[0,1], np.corrcoef(x, m.future_spend_4w)[0,1], np.corrcoef(x, m.oof)[0,1]))
print(m[['disp_share_84','mail_share_84','premium_idx','store_rich_84']].describe().round(4))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd, time, xgboost as xgb
t0=time.time()
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); cand = A.load_saved('cand_train.parquet')
train_days = sorted(tt.snapshot_day.unique())
key = ['household_key','snapshot_day']
fcols = [c for c in feats.columns if c not in key]
candc = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84']
tr = feats[feats.snapshot_day.isin(train_days)].merge(tt, on=key).merge(cand[key+candc], on=key, how='left')
print('tr', tr.shape, 'cand missing rows:', tr[candc[0]].isna().sum())
fill0 = ['units_84','spend84','trips_7','nprod_28','ncomm_28','disp_share_84','mail_share_84']
for c in fill0: tr[c] = tr[c].fillna(0.0)
tr['up_mean_84'] = tr.up_mean_84.fillna(0.0); tr['premium_idx'] = tr.premium_idx.fillna(1.0)
tr['store_rich_84'] = tr.store_rich_84.fillna(tr.store_rich_84.median())
y = tr.future_spend_4w.values
Xb = tr[fcols].values.astype(np.float32)
Xn = tr[fcols+candc+['up_mean_84','premium_idx','store_rich_84']].values.astype(np.float32)
print('Xb', Xb.shape, 'Xn', Xn.shape)
CONFIGS = [dict(max_depth=4,min_child_weight=20), dict(max_depth=5,min_child_weight=40),
           dict(max_depth=4,min_child_weight=40), dict(max_depth=6,min_child_weight=60)]
def bag(Xtr, ytr, Xte):
    p = np.zeros(len(Xte))
    for c in CONFIGS:
        prm = {'objective':'reg:quantileerror','quantile_alpha':0.5,'learning_rate':0.08,
               'n_estimators':400,'nthread':4,'seed':0, 'base_score':float(np.median(ytr))}
        prm.update(c)
        m = xgb.XGBRegressor(**prm); m.fit(Xtr, ytr)
        p += m.predict(Xte)
    return p/len(CONFIGS)
days = tr.snapshot_day.values
def loso(X):
    oof = np.zeros(len(tr))
    for d in train_days:
        te = days==d
        oof[te] = bag(X[~te], y[~te], X[te])
    return oof
t1=time.time(); oof_b = loso(Xb); print('arm a (base) LOSO MAE %.4f  [%.0fs]' % (np.abs(oof_b-y).mean(), time.time()-t1))
t1=time.time(); oof_n = loso(Xn); print('arm b (+cand) LOSO MAE %.4f  [%.0fs]' % (np.abs(oof_n-y).mean(), time.time()-t1))
med = np.median(y)
for gname in ['wk_cv8','wk_std8']:
    gvar = tr[gname].values
    q = np.quantile(gvar, [1/3, 2/3]); grp = np.digitize(gvar, q)
    pred = oof_b.copy()
    for te_day in train_days:
        te = days==te_day
        for g in range(3):
            m_te = te & (grp==g); m_tr = (~te) & (grp==g)
            if m_tr.sum()<100: continue
            ss = np.linspace(0.5,1.0,11)
            maes = [np.abs(med + s*(oof_b[m_tr]-med) - y[m_tr]).mean() for s in ss]
            s_star = ss[int(np.argmin(maes))]
            pred[m_te] = med + s_star*(oof_b[m_te]-med)
    print('gated shrink by %s: CV MAE %.4f' % (gname, np.abs(pred-y).mean()))
print('total %.0fs' % (time.time()-t0))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd, time, xgboost as xgb
t0=time.time()
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); cand = A.load_saved('cand_train.parquet')
train_days = sorted(tt.snapshot_day.unique())
key = ['household_key','snapshot_day']
fcols = [c for c in feats.columns if c not in key]
demo_cols = [c for c in fcols if feats[c].dtype==object]
print('string cols:', demo_cols)
def enc(df):
    df = df.copy()
    for c in demo_cols: df[c] = pd.Categorical(df[c].astype(str)).codes
    return df
feats = enc(feats)
candc = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84']
tr = feats[feats.snapshot_day.isin(train_days)].merge(tt, on=key).merge(cand[key+candc], on=key, how='left')
print('tr', tr.shape, 'cand missing:', tr[candc[0]].isna().sum())
for c in ['units_84','spend84','trips_7','nprod_28','ncomm_28','disp_share_84','mail_share_84','up_mean_84']: tr[c] = tr[c].fillna(0.0)
tr['premium_idx'] = tr.premium_idx.fillna(1.0)
tr['store_rich_84'] = tr.store_rich_84.fillna(tr.store_rich_84.median())
y = tr.future_spend_4w.values
Xb = tr[fcols].values.astype(np.float32)
Xn = tr[fcols+candc].values.astype(np.float32)
print('Xb', Xb.shape, 'Xn', Xn.shape)
CONFIGS = [dict(max_depth=4,min_child_weight=20), dict(max_depth=5,min_child_weight=40),
           dict(max_depth=4,min_child_weight=40), dict(max_depth=6,min_child_weight=60)]
def bag(Xtr, ytr, Xte):
    p = np.zeros(len(Xte))
    for c in CONFIGS:
        prm = {'objective':'reg:quantileerror','quantile_alpha':0.5,'learning_rate':0.08,
               'n_estimators':400,'nthread':4,'seed':0, 'base_score':float(np.median(ytr))}
        prm.update(c)
        m = xgb.XGBRegressor(**prm); m.fit(Xtr, ytr)
        p += m.predict(Xte)
    return p/len(CONFIGS)
days = tr.snapshot_day.values
def loso(X):
    oof = np.zeros(len(tr))
    for d in train_days:
        te = days==d
        oof[te] = bag(X[~te], y[~te], X[te])
    return oof
t1=time.time(); oof_b = loso(Xb); print('arm a (base) LOSO MAE %.4f  [%.0fs]' % (np.abs(oof_b-y).mean(), time.time()-t1))
t1=time.time(); oof_n = loso(Xn); print('arm b (+cand) LOSO MAE %.4f  [%.0fs]' % (np.abs(oof_n-y).mean(), time.time()-t1))
med = np.median(y)
for gname in ['wk_cv8','wk_std8']:
    gvar = tr[gname].values
    q = np.quantile(gvar, [1/3, 2/3]); grp = np.digitize(gvar, q)
    pred = oof_b.copy()
    for te_day in train_days:
        te = days==te_day
        for g in range(3):
            m_te = te & (grp==g); m_tr = (~te) & (grp==g)
            if m_tr.sum()<100: continue
            ss = np.linspace(0.5,1.0,11)
            maes = [np.abs(med + s*(oof_b[m_tr]-med) - y[m_tr]).mean() for s in ss]
            s_star = ss[int(np.argmin(maes))]
            pred[m_te] = med + s_star*(oof_b[m_te]-med)
    print('gated shrink by %s: CV MAE %.4f' % (gname, np.abs(pred-y).mean()))
print('total %.0fs' % (time.time()-t0))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); cand = A.load_saved('cand_train.parquet')
print('feats dtypes non-numeric:')
for c in feats.columns:
    if not np.issubdtype(feats[c].dtype, np.number): print('  ', c, feats[c].dtype)
print('tt days', sorted(tt.snapshot_day.unique()))
print('cand days', sorted(cand.snapshot_day.unique()))
print('cand sizes:'); print(cand.snapshot_day.value_counts().sort_index())
print('tt sizes:'); print(tt.snapshot_day.value_counts().sort_index())
k1 = set(map(tuple, tt[['household_key','snapshot_day']].values))
k2 = set(map(tuple, cand[['household_key','snapshot_day']].values))
print('tt pairs', len(k1), 'cand pairs', len(k2), 'tt-not-in-cand', len(k1-k2), 'cand-not-in-tt', len(k2-k1))
print('dup cand pairs:', cand.duplicated(['household_key','snapshot_day']).sum())


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd, time, xgboost as xgb
t0=time.time()
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); oof = A.load_saved('oof_e013.parquet')
cand = A.load_saved('cand_train.parquet')
key = ['household_key','snapshot_day']
tr = tt.merge(oof, on=key).merge(cand, on=key, how='left').merge(feats[key+['wk_cv8','wk_std8','spend_28','exp4w_blend']], on=key, how='left')
print('tr', tr.shape, 'cand missing:', tr.units_84.isna().sum())
for c in ['units_84','spend84','trips_7','nprod_28','ncomm_28','disp_share_84','mail_share_84','up_mean_84']: tr[c] = tr[c].fillna(0.0)
tr['premium_idx'] = tr.premium_idx.fillna(1.0); tr['store_rich_84'] = tr.store_rich_84.fillna(tr.store_rich_84.median())
y = tr.future_spend_4w.values; base = tr.oof.values; r = y - base
days = tr.snapshot_day.values; tdays = sorted(tt.snapshot_day.unique())
print('resid: mean %.2f median %.2f std %.1f' % (r.mean(), np.median(r), r.std()))
def mae(p): return np.abs(p-y).mean()
print('base OOF MAE %.4f' % mae(base))
# 1) residual model g(cand), honest LOSO
candc = ['units_84','spend84','trips_7','nprod_28','ncomm_28','up_mean_84','premium_idx','store_rich_84','disp_share_84','mail_share_84']
Xc = tr[candc].values.astype(np.float32)
oof_g = np.zeros(len(tr))
for d in tdays:
    te = days==d
    m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=3, min_child_weight=50,
                         learning_rate=0.1, n_estimators=150, nthread=4, seed=0)
    m.fit(Xc[~te], r[~te]); oof_g[te] = m.predict(Xc[te])
print('g std on oof %.3f' % oof_g.std())
for w in [0.25,0.5,1.0]:
    print('base + %.2f*g(cand): LOSO MAE %.4f' % (w, mae(base + w*oof_g)))
# 2) constant shift
print('base+median(r): %.4f | base+0.5*mean(r): %.4f' % (mae(base+np.median(r)), mae(base+0.5*r.mean())))
# 3) clip tests
for c in [300,400,500,600,800,1000]:
    print('clip@%d: %.4f' % (c, mae(np.minimum(base,c))))
for q in [0.98,0.99,0.995]:
    c = np.quantile(base,q); print('clip@q%.3f(=%.0f): %.4f' % (q,c,mae(np.minimum(base,c))))
# 4) gated shrink by wk_cv8 terciles (honest over days)
for gname in ['wk_cv8','wk_std8']:
    gvar = tr[gname].values; q3 = np.nanquantile(gvar,[1/3,2/3]); grp = np.digitize(gvar,q3)
    pred = base.copy()
    for d in tdays:
        te = days==d
        for g in range(3):
            mte = te&(grp==g); mtr = (~te)&(grp==g)
            if mtr.sum()<100 or mte.sum()==0: continue
            ss = np.linspace(0.6,1.0,9)
            s = ss[int(np.argmin([np.abs(np.median(r[mtr]) + s*(base[mtr]-np.median(r[mtr])) - y[mtr]).mean() for s in ss]))]
            pred[mte] = np.median(r[mtr]) + s*(base[mte]-np.median(r[mtr]))
    print('gated shrink by %s: %.4f' % (gname, mae(pred)))
print('total %.0fs' % (time.time()-t0))


# ---- cell ----

import agent_api as A, numpy as np, pandas as pd, time, xgboost as xgb
t0=time.time()
feats = A.load_saved('feats_v4.parquet'); tt = A.train_targets(); oof = A.load_saved('oof_e013.parquet')
key = ['household_key','snapshot_day']
# encode any categorical columns to integer codes
for c in feats.columns:
    if str(feats[c].dtype) == 'category' or feats[c].dtype == object:
        feats[c] = pd.factorize(feats[c].astype(str))[0]
val_days = [459,487,515,543]
trm = feats.snapshot_day.isin(sorted(tt.snapshot_day.unique()))
vm = feats.snapshot_day.isin(val_days)
tr = feats[trm].merge(tt, on=key)
print('train rows', len(tr), 'val rows', vm.sum())
y = tr.future_spend_4w.values
fcols = [c for c in feats.columns if c not in key]
Xtr = tr[fcols].values.astype(np.float32)
Xva = feats[vm][fcols].values.astype(np.float32)
CONFIGS = [(4,20),(5,40),(4,40),(6,60)]
med_y = float(np.median(y))
pv = np.zeros(len(Xva))
for seed in (0,1):
    for md, mcw in CONFIGS:
        m = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.5, max_depth=md,
                             min_child_weight=mcw, learning_rate=0.08, n_estimators=400,
                             nthread=4, seed=seed, base_score=med_y)
        m.fit(Xtr, y)
        pv += m.predict(Xva)
pv /= (2*len(CONFIGS))
# honest global shift: median train-OOF residual
mg = tr[key+['future_spend_4w']].merge(oof, on=key)
shift = float(np.median(mg.future_spend_4w.values - mg.oof.values))
print('shift = %.3f' % shift)
pred = np.clip(pv + shift, 0, None)
out = feats.loc[vm, key].copy(); out['prediction'] = pred
print('out', out.shape, 'nan:', out.prediction.isna().sum(), 'mean %.2f median %.2f' % (out.prediction.mean(), out.prediction.median()))
print('val pred mean by day:'); print(out.groupby('snapshot_day').prediction.mean().round(2))
assert len(out)==9989 and out.prediction.notna().all()
A.save_table(out, 'pred_e020.parquet')
print('saved OK, elapsed %.0fs' % (time.time()-t0))
