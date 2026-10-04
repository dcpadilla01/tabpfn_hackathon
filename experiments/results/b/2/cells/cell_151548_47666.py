import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def analog_feats(view, snapshot_day):
    day = int(snapshot_day)
    tx = view.transactions[['household_key','day','sales_value','basket_id']]
    hh = pd.Index(view.households if isinstance(view.households, pd.Index) else np.asarray(view.households).ravel())
    # compact feature builder for any anchor day <= day
    def build(anchor):
        t = tx[tx.day <= anchor]
        g = t.groupby('household_key')['sales_value']
        f = pd.DataFrame(index=hh)
        f['s28'] = t[t.day > anchor-28].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s84'] = t[t.day > anchor-84].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['s364'] = t[t.day > anchor-364].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        f['rec'] = (anchor - t.groupby('household_key')['day'].max()).reindex(hh).fillna(999)
        f['ten'] = (anchor - t.groupby('household_key')['day'].min()).reindex(hh).fillna(0)
        f['ntr28'] = t[t.day > anchor-28].groupby('household_key')['basket_id'].nunique().reindex(hh).fillna(0)
        # zero streak in trailing 28d blocks
        sp = [t[(t.day > anchor-28*k) & (t.day <= anchor-28*(k-1))].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0).values for k in range(1,5)]
        zs = np.zeros(len(hh))
        for i in range(len(hh)):
            for k in range(4):
                if sp[k][i] <= 0: zs[i] += 1
                else: break
        f['zs'] = zs
        return f
    # training rows: past snapshots with realized future 4w spend (needs anchor+28 <= day)
    rows, ys = [], []
    anchors = [day - 28*k for k in range(1, 13) if day - 28*k >= 95]
    for a in anchors:
        f = build(a)
        fut = tx[(tx.day > a) & (tx.day <= a+28)].groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0)
        rows.append(f); ys.append(fut.values)
    Xtr = pd.concat(rows); ytr = np.concatenate(ys)
    Xcur = build(day)
    # k-NN on log features
    cols = ['s28','s84','s364','rec','ten','ntr28','zs']
    Ltr = np.log1p(Xtr[cols].clip(lower=0).values)
    Lcu = np.log1p(Xcur[cols].clip(lower=0).values)
    mu,sd = Ltr.mean(0), Ltr.std(0); sd[sd==0]=1
    A = (Ltr-mu)/sd; B = (Lcu-mu)/sd
    yl = np.log1p(ytr)
    # subsample for speed
    rng = np.random.RandomState(0)
    idx = rng.choice(len(A), min(len(A), 12000), replace=False)
    A, yl_s = A[idx], yl[idx]
    # chunked k-NN, k=25
    k = 25
    preds = np.zeros(len(B))
    for st in range(0, len(B), 2000):
        en = min(st+2000, len(B))
        d = -2*A@B[st:en].T + (A*A).sum(1)[:,None]
        nn = np.argpartition(d, k, axis=0)[:k]
        preds[st:en] = np.expm1(yl_s[nn].mean(0))
    Xcur['knn28'] = preds
    # binned conditional mean: (s28 decile) x recency bucket
    qb = pd.qcut(np.log1p(Xtr['s28']), 10, labels=False, duplicates='drop')
    rb = pd.cut(Xtr['rec'], [-1,7,14,28,56,112,10**6], labels=False)
    key = pd.Series(qb.astype(str)+'_'+rb.astype(str))
    tab = pd.Series(yl).groupby(key.values).mean()
    kc = pd.Series(pd.qcut(np.log1p(Xcur['s28']), 10, labels=False, duplicates='drop').astype(str)+'_'+
                   pd.cut(Xcur['rec'], [-1,7,14,28,56,112,10**6], labels=False).astype(str))
    Xcur['binmean'] = kc.map(tab).astype(float).fillna(yl.mean()).values
    Xcur['log_knn'] = np.log1p(preds.clip(lower=0))
    Xcur['log_bin'] = np.log1p(Xcur['binmean'].clip(lower=0))
    return Xcur[['knn28','binmean','log_knn','log_bin']].astype(float)

bf = agent_api.build_features(analog_feats)
print('built', bf.shape)
e008 = agent_api.load_saved('e008_fwd_calendar.parquet')
comb = e008.merge(bf, on=['household_key','snapshot_day'], how='inner')
print('combined', comb.shape)

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
    return np.abs(p-yva).mean()

print('proxy E008          val431: %.3f' % proxy_mae(e008,[431]))
print('proxy E008+analog   val431: %.3f' % proxy_mae(comb,[431]))
print('proxy E008          v403+431: %.3f' % proxy_mae(e008,[403,431]))
print('proxy E008+analog   v403+431: %.3f' % proxy_mae(comb,[403,431]))
# analog features alone
an = tt.merge(bf, on=['household_key','snapshot_day'])
print('proxy analog-only   val431: %.3f' % proxy_mae(bf,[431]))
# correlation with target
mm = tt.merge(bf, on=['household_key','snapshot_day'])
print('\ncorr with target:'); print(mm[['knn28','binmean','log_knn','log_bin','future_spend_4w']].corr()['future_spend_4w'].round(3))
agent_api.save_table(comb, 'e009_analog.parquet')