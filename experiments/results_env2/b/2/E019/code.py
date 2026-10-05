
import pandas as pd, numpy as np
e011 = load_saved('e011_table.parquet')
print('e011', e011.shape)
print(e011.columns.tolist())
tt = train_targets()
print('targets', tt.shape)
print(tt.groupby('snapshot_day')['future_spend_4w'].agg([('mean','mean'),('med','median'),('zero',lambda s:(s==0).mean())]))
for name in ['deal_v1','hazard_v1','timing_v1','selfcal_v1','rfm_traj_v1','season_demo_v1','twin_v1','display_v1','lvl_v1','union_all_v1','combined_v1','e013_table','churn_vol_v1']:
    try:
        df = load_saved(name + '.parquet')
        print('---', name, df.shape)
        print(df.columns.tolist())
    except Exception as e:
        print('---', name, 'ERR', type(e).__name__)


# ---- cell ----

import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
print(e.dtypes.value_counts())
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner')
print('merged', df.shape)
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
print('nan frac total', X.isna().mean().mean())
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
tr = ~np.isin(days, [403,431]); va = np.isin(days, [403,431])
print('local train', tr.sum(), 'local val', va.sum())

def ridge_eval(cols, alphas=(3,10,30,100,300,1000)):
    Xm = X[cols].fillna(0.0).values
    mu = Xm[tr].mean(0); sd = Xm[tr].std(0)+1e-9
    Z = (Xm-mu)/sd
    Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[va], np.ones(va.sum())]
    ytr = y[tr]
    out=[]
    for a in alphas:
        A = Ztr.T@Ztr + a*np.eye(Ztr.shape[1]); A[-1,-1]-=a
        w = np.linalg.solve(A, Ztr.T@ytr)
        pred = Zva@w
        out.append((a, np.abs(pred-y[va]).mean()))
    return out

base = ridge_eval(feats)
print('E011 all:', base)
# core subset
core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
print('core:', ridge_eval(core))


# ---- cell ----

import pandas as pd, numpy as np
for name in ['deal_v1','hazard_v1','timing_v1','twin_v1','display_v1','lvl_v1','mkt_v1','comp_v1','ewma_block_v1','selfcal_v1','union_all_v1','combined_v1','e017_union_v1','e014_table','e014_retry']:
    try:
        df = load_saved(name+'.parquet')
        extra = [c for c in df.columns if c not in ('household_key','snapshot_day')]
        print('---', name, df.shape, 'n_extra', len(extra))
        print(extra[:40])
    except Exception as e:
        print('---', name, 'ERR', type(e).__name__, e)


# ---- cell ----

import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner')
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
tr = ~np.isin(days, [403,431]); va = np.isin(days, [403,431])
mu = X[tr].mean(0); sd = X[tr].std(0)+1e-9
Z = ((X-mu)/sd).fillna(0.0).values
Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[va], np.ones(va.sum())]
ytr = y[tr]
A = Ztr.T@Ztr + 1000*np.eye(Ztr.shape[1]); A[-1,-1]-=1000
w = np.linalg.solve(A, Ztr.T@ytr)
res = Zva@w - y[va]
print('baseline local MAE', np.abs(res).mean())
# top |corr| features with residual
Zv = Z[va][:,:len(feats)]
rv = res
cs = []
for j,f in enumerate(feats):
    zj = Zv[:,j]
    s = zj.std()
    if s < 1e-9: continue
    c = np.corrcoef(zj, rv)[0,1]
    cs.append((abs(c), c, f))
cs.sort(reverse=True)
for a,c,f in cs[:35]: print(f'{f:28s} r={c:+.3f}')


# ---- cell ----

import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key'])
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
X = df[feats].apply(pd.to_numeric, errors='coerce')
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
tr_days = [95,123,151,179,207,235,263,291,319,347,375,403,431]

def loso_mae(cols, alpha=1000, pairs=None):
    Xm = X[list(cols)].fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    if pairs is None:
        pairs = [((403,431),), ((151,179),), ((263,291),), ((319,347),)]
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair))
        tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes

# candidate extra blocks
deal = load_saved('deal_v1.parquet'); haz = load_saved('hazard_v1.parquet'); tim = load_saved('timing_v1.parquet'); disp = load_saved('display_v1.parquet')
for t in (deal,haz,tim,disp):
    t.sort_values(['snapshot_day','household_key'], inplace=True)
assert (deal.household_key.values==df.household_key.values).all() and (deal.snapshot_day.values==df.snapshot_day.values).all()

blocks = {
 'deal': ['deal_share_84','coup_share_84','deal_line_frac_84','deep_deal_share_84','coup_trips_84','redemp_84','deal_share_364','coup_share_364','deal_line_frac_364','deep_deal_share_364','coup_trips_364','redemp_364','deal_trend'],
 'haz': ['h_gap_mean','h_gap_med','h_gap_std','h_gap_max','h_gap_p90','h_n_gaps','h_gap_last3','h_gap_ratio','h_days_since','h_hazard','h_zero_stretch_112','h_trips_7','h_trips_14','h_active_frac_84'],
 'tim': ['t_hour_mean','t_hour_std','t_morning_sp_share','t_evening_sp_share','t_dow_hhi','t_n_dow','t_n_stores','t_store_share'],
 'disp': ['disp_share_84','mail_share_84','mailAD_share_84','disp_line_frac_84','disp_share_364','mail_share_364','mailAD_share_364','disp_line_frac_364','disp_trend'],
}
m0,_ = loso_mae(feats); print('E011 all %.3f' % m0)
core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
m1,_ = loso_mae(core); print('core48 %.3f' % m1)
for bn, bc in blocks.items():
    m,_ = loso_mae(core+bc); print(f'core48+{bn} %.3f' % m)
mall,_ = loso_mae(core+deal+haz+tim+disp, )


# ---- cell ----

import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
days = df['snapshot_day'].values

blocks = {}
deal = load_saved('deal_v1.parquet'); haz = load_saved('hazard_v1.parquet'); tim = load_saved('timing_v1.parquet'); disp = load_saved('display_v1.parquet')
for nm,t in [('deal',deal),('haz',haz),('tim',tim),('disp',disp)]:
    t = t.merge(df[['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
    assert len(t)==len(df), (nm, len(t))
    assert (t.household_key.values==df.household_key.values).all()
    blocks[nm] = t

X = df[feats].apply(pd.to_numeric, errors='coerce')

def loso_mae(Xdf, cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = Xdf[list(cols)].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair)); tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes

core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
m0,_ = loso_mae(df, feats); print('E011 all161 %.3f' % m0)
m1,_ = loso_mae(df, core); print('core48 %.3f' % m1)
for bn in blocks:
    bc = [c for c in blocks[bn].columns if c not in ('household_key','snapshot_day')]
    d2 = pd.concat([df[['household_key','snapshot_day']], blocks[bn][bc]], axis=1)
    m,_ = loso_mae(d2, core+bc); print(f'core48+{bn}({len(bc)}) %.3f' % m)
# all extras together
allb = pd.concat([df[['household_key','snapshot_day']]] + [blocks[b][[c for c in blocks[b].columns if c not in ('household_key','snapshot_day')]] for b in blocks], axis=1)
m,_ = loso_mae(allb, core + sum([[c for c in blocks[b].columns if c not in ('household_key','snapshot_day')] for b in blocks], []))
print('core48+all4blocks %.3f' % m)


# ---- cell ----

import pandas as pd, numpy as np
e = load_saved('e011_table.parquet')
tt = train_targets()
df = e.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in e.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
deal = load_saved('deal_v1.parquet'); haz = load_saved('hazard_v1.parquet'); tim = load_saved('timing_v1.parquet'); disp = load_saved('display_v1.parquet')
blocks={}
for nm,t in [('deal',deal),('haz',haz),('tim',tim),('disp',disp)]:
    t = t.merge(df[['household_key','snapshot_day']], on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
    assert (t.household_key.values==df.household_key.values).all()
    bc = [c for c in t.columns if c not in ('household_key','snapshot_day')]
    blocks[nm]=t[bc]

def loso_mae(Xdf, cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = Xdf[list(cols)].apply(pd.to_numeric, errors='coerce').fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair)); tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes

core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
m0,_ = loso_mae(df[feats], feats); print('E011 all161 %.3f' % m0)
m1,_ = loso_mae(df[feats], core); print('core48 %.3f' % m1)
base = df[feats]
for bn in blocks:
    d2 = pd.concat([base, blocks[bn]], axis=1)
    m,_ = loso_mae(d2, core+list(blocks[bn].columns)); print(f'core48+{bn}({blocks[bn].shape[1]}) %.3f' % m)
allb = pd.concat([base]+list(blocks.values()), axis=1)
m,_ = loso_mae(allb, core+sum([list(b.columns) for b in blocks.values()],[]))
print('core48+all4blocks %.3f' % m)
# also full E011 + all 4 blocks
m,_ = loso_mae(allb, feats+sum([list(b.columns) for b in blocks.values()],[]))
print('E011+all4blocks %.3f' % m)


# ---- cell ----

import pandas as pd, numpy as np

NEWCOLS = ['x_spend_7d','x_spend_14d','x_trips_7d','x_trips_14d','x_lvl_ew2','x_f_ew_hl6','x_f_ew_hl8',
           'x_f56_mean','x_f84_mean','x_r_std','x_r_ew_hl8','x_zero_streak','x_active_frac13',
           'x_p_s28_rmean','x_p_s28_rmed','x_p_lvl_rmean','x_p_s28_fmean','x_p_usual_rmean','x_p_s28_carry','x_p_ya_rmean']

def fn(view, snapshot_day):
    s = int(snapshot_day)
    hh = pd.Index(np.asarray(view.households).ravel())
    tx = view.table('transactions')
    tx = tx[tx['household_key'].isin(hh)]
    ag = tx.groupby(['household_key','day']).agg(sp=('sales_value','sum'), nb=('basket_id','nunique')).reset_index()
    full = list(range(1, s+1))
    spm = ag.pivot(index='household_key', columns='day', values='sp').reindex(index=hh, columns=full).fillna(0.0)
    nbm = ag.pivot(index='household_key', columns='day', values='nb').reindex(index=hh, columns=full).fillna(0.0)
    Sv = spm.values; Nv = nbm.values; n = len(hh)
    Kmax = max(1, (s-28)//28 + 1)
    W = np.empty((n, Kmax))
    for k in range(Kmax):
        lo = max(0, s-28*(k+1)); hi = s-28*k
        W[:,k] = Sv[:, lo:hi].sum(1)
    s7 = Sv[:, max(0,s-7):s].sum(1); s14 = Sv[:, max(0,s-14):s].sum(1)
    t7 = Nv[:, max(0,s-7):s].sum(1); t14 = Nv[:, max(0,s-14):s].sum(1)
    fd = tx.groupby('household_key')['day'].min().reindex(hh).values.astype(float)
    Kh = np.clip(np.floor((s - fd - 27)/28).astype(int) + 1, 1, Kmax)
    out = {c: np.full(n, np.nan) for c in NEWCOLS}
    w2 = 0.5**(np.arange(Kmax)/2.0); w6 = 0.5**(np.arange(Kmax)/6.0); w8 = 0.5**(np.arange(Kmax)/8.0)
    for i in range(n):
        K = int(Kh[i]); seg = W[i,:K]; w0 = seg[0]
        lvl = (seg*w2[:K]).sum()/w2[:K].sum()
        f6 = (seg*w6[:K]).sum()/w6[:K].sum(); f8 = (seg*w8[:K]).sum()/w8[:K].sum()
        f56 = (seg[:-1]+seg[1:]).mean() if K>=2 else np.nan
        f84 = (seg[:-2]+seg[1:-1]+seg[2:]).mean() if K>=3 else np.nan
        r_mean=r_med=r_std=r_ew=np.nan
        m = seg[1:] > 0
        if m.any():
            r = seg[:-1][m]/seg[1:][m]
            r_mean=r.mean(); r_med=np.median(r); r_std=r.std()
            kk = np.arange(1,K)[m]; ww = 0.5**((kk-1)/8.0); r_ew=(r*ww).sum()/ww.sum()
        z=0
        while z<K and seg[z]==0: z+=1
        act = (seg[:min(13,K)]>0).mean()
        usual = np.median(seg[1:min(K,14)]) if K>=2 else np.nan
        ya = seg[13] if K>13 else np.nan
        carry = ((seg[:-1]>0)&(seg[1:]>0)).mean()
        out['x_spend_7d'][i]=s7[i]; out['x_spend_14d'][i]=s14[i]
        out['x_trips_7d'][i]=t7[i]; out['x_trips_14d'][i]=t14[i]
        out['x_lvl_ew2'][i]=lvl; out['x_f_ew_hl6'][i]=f6; out['x_f_ew_hl8'][i]=f8
        out['x_f56_mean'][i]=f56; out['x_f84_mean'][i]=f84
        out['x_r_std'][i]=r_std; out['x_r_ew_hl8'][i]=r_ew
        out['x_zero_streak'][i]=z; out['x_active_frac13'][i]=act
        out['x_p_s28_rmean'][i]=w0*r_mean; out['x_p_s28_rmed'][i]=w0*r_med
        out['x_p_lvl_rmean'][i]=lvl*r_mean; out['x_p_s28_fmean'][i]=w0*seg.mean()
        out['x_p_usual_rmean'][i]=usual*r_mean; out['x_p_s28_carry'][i]=w0*carry
        out['x_p_ya_rmean'][i]=ya*r_mean
    return pd.DataFrame(out, index=hh)

res = build_features(fn)
print('built', res.shape, sorted(res.snapshot_day.unique()))
print(res[NEWCOLS].isna().mean().round(3).to_dict())
e = load_saved('e011_table.parquet')
print('e011', e.shape)
comb = e.merge(res, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)
assert len(comb)==36426, len(comb)
assert comb.duplicated(['household_key','snapshot_day']).sum()==0
p = save_table(comb, 'e019_table')
print('saved', p)


# ---- cell ----

import pandas as pd, numpy as np
comb = load_saved('e019_table.parquet')
tt = train_targets()
df = comb.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in comb.columns if c not in ('household_key','snapshot_day')]
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
X = df[feats].apply(pd.to_numeric, errors='coerce')

def loso_mae(cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = X[list(cols)].fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair)); tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes

NEW = ['x_spend_7d','x_spend_14d','x_trips_7d','x_trips_14d','x_lvl_ew2','x_f_ew_hl6','x_f_ew_hl8',
       'x_f56_mean','x_f84_mean','x_r_std','x_r_ew_hl8','x_zero_streak','x_active_frac13',
       'x_p_s28_rmean','x_p_s28_rmed','x_p_lvl_rmean','x_p_s28_fmean','x_p_usual_rmean','x_p_s28_carry','x_p_ya_rmean']
m0,_ = loso_mae(feats); print('E011 all %.3f' % m0)
m1,_ = loso_mae(feats+NEW); print('E011+NEW20 %.3f' % m1)
# NEW alone
m2,_ = loso_mae(NEW); print('NEW20 alone %.3f' % m2)
# core + NEW
core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','trips_w1','trips_w2','usual_4w','ratio_recent_usual','tenure_days','gap_mean','gap_std','wk_spend_mean','wk_spend_std','ew_spend_hl14','ew_spend_hl28','ew_spend_hl56','ew_spend_hl112','ya_spend','ya_trips','ya_cov','ratio_ya_28','b_life_spend','b_w_mean6','b_w_cv6','sc_f_mean','sc_f_med','sc_f_ew_hl2','sc_f_ew_hl4','sc_ratio_mean','sc_carry','sc_carry_act','sc_f_zero_frac','sc_f_active_mean','has_demo']
m3,_ = loso_mae(core+NEW); print('core48+NEW20 %.3f' % m3)


# ---- cell ----

import pandas as pd, numpy as np
comb = load_saved('e019_table.parquet')
tt = train_targets()
df = comb.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in comb.columns if c not in ('household_key','snapshot_day')]
NEW = ['x_spend_7d','x_spend_14d','x_trips_7d','x_trips_14d','x_lvl_ew2','x_f_ew_hl6','x_f_ew_hl8',
       'x_f56_mean','x_f84_mean','x_r_std','x_r_ew_hl8','x_zero_streak','x_active_frac13',
       'x_p_s28_rmean','x_p_s28_rmed','x_p_lvl_rmean','x_p_s28_fmean','x_p_usual_rmean','x_p_s28_carry','x_p_ya_rmean']
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
X = df[feats].apply(pd.to_numeric, errors='coerce')
def loso_mae(cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = X[list(cols)].fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair)); tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes
core = ['spend_28d','spend_56d','spend_84d','spend_112d','spend_168d','spend_364d','trips_28d','trips_84d','trips_364d','days_since_last','avg_basket_112','trend_28','spend_w1','spend_w2','spend_w3','spend_w4','spend_w5','spend_w6','tr wrong']


# ---- cell ----

import pandas as pd, numpy as np
comb = load_saved('e019_table.parquet')
tt = train_targets()
df = comb.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in comb.columns if c not in ('household_key','snapshot_day')]
NEW = ['x_spend_7d','x_spend_14d','x_trips_7d','x_trips_14d','x_lvl_ew2','x_f_ew_hl6','x_f_ew_hl8',
       'x_f56_mean','x_f84_mean','x_r_std','x_r_ew_hl8','x_zero_streak','x_active_frac13',
       'x_p_s28_rmean','x_p_s28_rmed','x_p_lvl_rmean','x_p_s28_carry','x_p_s28_fmean','x_p_usual_rmean','x_p_ya_rmean']
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
X = df[feats].apply(pd.to_numeric, errors='coerce')
def loso_mae(cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = X[list(cols)].fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair)); tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes
base,_ = loso_mae(feats); print('E011 base %.3f' % base)
cands = NEW[:]
chosen = []
best = base
for it in range(6):
    scores = [(loso_mae(feats+chosen+[c])[0], c) for c in cands]
    scores.sort()
    m,c = scores[0]
    if m >= best - 0.01:
        break
    chosen = chosen + [c]
    cands = [x for x in cands if x != c]
    best = m
    print('added', c, '-> %.3f' % m)
print('chosen', chosen)


# ---- cell ----

import pandas as pd, numpy as np
comb = load_saved('e019_table.parquet')
tt = train_targets()
df = comb.merge(tt, on=['household_key','snapshot_day'], how='inner').sort_values(['snapshot_day','household_key']).reset_index(drop=True)
feats = [c for c in comb.columns if c not in ('household_key','snapshot_day')]
NEW = ['x_spend_7d','x_spend_14d','x_trips_7d','x_trips_14d','x_lvl_ew2','x_f_ew_hl6','x_f_ew_hl8',
       'x_f56_mean','x_f84_mean','x_r_std','x_r_ew_hl8','x_zero_streak','x_active_frac13',
       'x_p_s28_rmean','x_p_s28_rmed','x_p_lvl_rmean','x_p_s28_carry','x_p_s28_fmean','x_p_usual_rmean','x_p_ya_rmean']
y = df['future_spend_4w'].values
days = df['snapshot_day'].values
X = df[feats].apply(pd.to_numeric, errors='coerce')
def loso_mae(cols, alpha=1000, pairs=((403,431),(151,179),(263,291),(319,347))):
    Xm = X[list(cols)].fillna(0.0).values
    mu = Xm.mean(0); sd = Xm.std(0)+1e-9
    Z = (Xm-mu)/sd
    maes=[]
    for pair in pairs:
        held = np.isin(days, list(pair)); tr = ~held
        Ztr = np.c_[Z[tr], np.ones(tr.sum())]; Zva = np.c_[Z[held], np.ones(held.sum())]
        A = Ztr.T@Ztr + alpha*np.eye(Ztr.shape[1]); A[-1,-1]-=alpha
        w = np.linalg.solve(A, Ztr.T@y[tr])
        maes.append(np.abs(Zva@w - y[held]).mean())
    return np.mean(maes), maes
base,_ = loso_mae(feats); print('E011 base %.3f' % base)
# per-single-feature delta
rows=[]
for c in NEW:
    m,_ = loso_mae(feats+[c])
    rows.append((m-base, c))
rows.sort()
for d,c in rows: print(f'{c:18s} delta {d:+.3f}')
