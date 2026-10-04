import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
e002 = agent_api.load_saved('e002_marketing_v2.parquet')
e009a = agent_api.load_saved('e009_analog.parquet')   # knn28, binmean, log_knn, log_bin
e009s = agent_api.load_saved('e009_spline2p.parquet') # includes p_active, lvl_given_active, pred_2p
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']

def loo_mae(table, alpha=1.0, feats=None):
    f = feats or [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    maes = []
    for d in tr_days:
        tr = m[m.snapshot_day != d]; va = m[m.snapshot_day == d]
        Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
        ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
        mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
        A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
        w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
        p = B@w
        maes.append(np.abs(p-yva).mean())
    return np.mean(maes)

def loo_single(col, table):
    # single-feature ridge (with intercept) ~ scaled fit
    return loo_mae(table, alpha=1.0, feats=[col])

print('standalone single-feature LOO MAE:')
for col in ['spend_84','spend_28','fwd28_mean','knn28','binmean','pred_2p','p_active','lvl_given_active','ewma8']:
    src = e009s if col in e009s.columns else (e009a if col in e009a.columns else e008)
    if col in src.columns:
        print('  %-18s %.3f' % (col, loo_single(col, src)))

print('\nLOO comparisons:')
c02 = e008.merge(e002.drop(columns=[c for c in e002.columns if c in e008.columns and c not in ('household_key','snapshot_day')]), on=['household_key','snapshot_day'], how='inner')
print('E008            : %.3f' % loo_mae(e008, alpha=100))
print('E008+E002(mkt)  : %.3f' % loo_mae(c02, alpha=100))
# E008 + only the 4 analog features
c_a = e008.merge(e009a[['household_key','snapshot_day','knn28','binmean','log_knn','log_bin']], on=['household_key','snapshot_day'], how='inner')
print('E008+analog4    : %.3f' % loo_mae(c_a, alpha=100))
# E008 + only pred_2p family
c_p = e008.merge(e009s[['household_key','snapshot_day','p_active','lvl_given_active','pred_2p']], on=['household_key','snapshot_day'], how='inner')
print('E008+2p3        : %.3f' % loo_mae(c_p, alpha=100))
# curated small: top features + analog
top = ['spend_84','spend_112','spend_56','wk_avg_8','spend_168','fwd28_mean','x_life_rate_wk','spend_rate_life',
       'fwd28_median','spend_28','wk_avg_4','fwd28_max','fwd28_k1','spend_lag1','recency','tenure',
       'knn28','binmean','pred_2p','day_idx','week_of_year','sin1','cos1','sin2','cos2','month_idx']
cur = e009s[e009s.columns.intersection(['household_key','snapshot_day']+top)]
print('curated-small   : %.3f' % loo_mae(cur, alpha=100))
print('curated-small a1: %.3f' % loo_mae(cur, alpha=1))
# blend check: correlation of knn28 residual-with-E008-fit? approximate: partial corr
m = tt.merge(e009a, on=['household_key','snapshot_day']).merge(e008, on=['household_key','snapshot_day'], suffixes=('','_8'))
print('\nknn28 corr vs spend_84:', round(m['knn28'].corr(m['spend_84']),3))
print('pred_2p corr vs spend_84:', round(e009s[['pred_2p']].merge(tt,on=['household_key','snapshot_day'])['pred_2p'].corr(
    tt.merge(e008,on=['household_key','snapshot_day'])['spend_84']),3))