import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def regime_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value']]
    hh = view.households
    if isinstance(hh, pd.DataFrame): hh = list(hh.index)
    else: hh = list(pd.unique(pd.Series(np.asarray(hh).ravel())))
    out = pd.DataFrame(index=hh)
    # trailing window spends
    for w in [28,56,84,112,168,364]:
        s = tx[tx.day > day-w].groupby('household_key')['sales_value'].sum()
        out['spend_%d'%w] = s.reindex(hh).fillna(0.0)
    # weekly spends wk0..wk7
    for k in range(8):
        s = tx[(tx.day > day-7*(k+1)) & (tx.day <= day-7*k)].groupby('household_key')['sales_value'].sum()
        out['w%d'%k] = s.reindex(hh).fillna(0.0)
    out['spend_life'] = tx.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)
    out['recency'] = (day - tx.groupby('household_key')['day'].max()).reindex(hh)
    out['tenure'] = (day - tx.groupby('household_key')['day'].min()).reindex(hh)
    # consecutive trailing 28d zero blocks (most recent first)
    zk = np.zeros(len(hh))
    sp = {}
    for k in range(1,7):
        s = tx[(tx.day > day-28*k) & (tx.day <= day-28*(k-1))].groupby('household_key')['sales_value'].sum()
        sp[k] = s.reindex(hh).fillna(0.0).values
    zc = np.zeros(len(hh))
    for i in range(len(hh)):
        for k in range(1,7):
            if sp[k][i] <= 0: zc[i] += 1
            else: break
    out['zero_streak28'] = zc
    # churn flags
    out['z28'] = (out['spend_28']<=0).astype(float)
    out['z56'] = (out['spend_56']<=0).astype(float)
    out['z84'] = (out['spend_84']<=0).astype(float)
    # recency bucket one-hots
    r = out['recency'].values
    out['r_le7']   = (r<=7).astype(float)
    out['r_8_14']  = ((r>7)&(r<=14)).astype(float)
    out['r_15_28'] = ((r>14)&(r<=28)).astype(float)
    out['r_29_56'] = ((r>28)&(r<=56)).astype(float)
    out['r_57_112']= ((r>56)&(r<=112)).astype(float)
    out['r_gt112'] = (r>112).astype(float)
    # log1p compressions
    for c in ['spend_28','spend_56','spend_84','spend_112','spend_168','spend_364','spend_life',
              'w0','w1','w2','w3','wk_avg_4','wk_avg_8','wk_avg_84','fwd28_mean','fwd28_median',
              'fwd28_max','fwd28_min','spend_lag1','spend_lag2','x_s28','x_prev28',
              'x_life_rate_wk','spend_rate_life','x_b_mean_val','x_b_med_val']:
        if c in out.columns:
            out['L_'+c] = np.log1p(out[c].clip(lower=0))
    # regime interactions: log level x recency regime
    L84 = out['L_spend_84'].values
    for b in ['r_le7','r_8_14','r_15_28','r_29_56','r_57_112','r_gt112']:
        out['i_'+b] = out[b].values * L84
    out['i_z28_L84'] = out['z28'].values * L84
    out['i_z28_L28'] = out['z28'].values * out['L_spend_28'].values
    out['i_streak_L84'] = out['zero_streak28'].values * L84
    # trend-in-log
    out['L_ratio_28_84'] = np.log1p(out['spend_28']) - np.log1p(out['spend_84'])
    out['L_ratio_84_364'] = np.log1p(out['spend_84']) - np.log1p(out['spend_364'].clip(lower=0))
    return out.astype(float)

bf = agent_api.build_features(regime_feats)
print('built', bf.shape, 'snapshots', sorted(bf.snapshot_day.unique()))

e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
newcols = [c for c in bf.columns if c not in ('household_key','snapshot_day')]
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)

# proxy eval helper
tt = agent_api.train_targets(); tr_days = agent_api.snapshot_days()['train']
def proxy_mae(table, val_days, alpha=1.0):
    f = [c for c in table.columns if c not in ('household_key','snapshot_day')]
    m = tt.merge(table, on=['household_key','snapshot_day'], how='inner')
    trd = [d for d in tr_days if d not in val_days]
    tr = m[m.snapshot_day.isin(trd)]; va = m[m.snapshot_day.isin(val_days)]
    Xtr = tr[f].astype(float).fillna(0).values; Xva = va[f].astype(float).fillna(0).values
    ytr = tr['future_spend_4w'].values; yva = va['future_spend_4w'].values
    mu,sd = Xtr.mean(0), Xtr.std(0); sd[sd==0]=1
    A = np.hstack([(Xtr-mu)/sd, np.ones((len(Xtr),1))]); B = np.hstack([(Xva-mu)/sd, np.ones((len(Xva),1))])
    w = np.linalg.solve(A.T@A+alpha*np.eye(A.shape[1]), A.T@ytr)
    p = B@w
    return np.abs(p-yva).mean(), np.abs(p-yva).mean()  # mae

base = proxy_mae(e008, [431]); print('proxy E008 val431: %.3f' % base[0])
c1 = proxy_mae(comb, [431]); print('proxy E008+regime val431: %.3f' % c1[0])
c2 = proxy_mae(comb, [403,431]); print('proxy E008+regime val403+431: %.3f' % c2[0])
b2 = proxy_mae(e008, [403,431]); print('proxy E008 val403+431: %.3f' % b2[0])

# also test: only logs+flags, no interactions
sub1 = e008.merge(bf[[c for c in newcols if not c.startswith('i_')]], on=['household_key','snapshot_day'])
print('proxy E008+logs/flags only val431: %.3f' % proxy_mae(sub1,[431])[0])

path = agent_api.save_table(comb, 'e009_regime.parquet')
print('saved', path)