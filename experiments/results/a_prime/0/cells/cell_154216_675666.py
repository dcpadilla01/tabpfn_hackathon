import numpy as np, pandas as pd

def rank_block(view, day):
    d = day
    hh = pd.Index(view.households)
    t = view.transactions[['household_key','basket_id','day','sales_value']]
    m28 = (t.day > d-28) & (t.day <= d)
    m84 = (t.day > d-84) & (t.day <= d)
    m364 = (t.day > d-364) & (t.day <= d)
    g28 = t[m28].groupby('household_key').agg(s28=('sales_value','sum'), b28=('basket_id','nunique'), days28=('day','nunique'))
    g84 = t[m84].groupby('household_key').agg(s84=('sales_value','sum'), b84=('basket_id','nunique'))
    g364 = t[m364].groupby('household_key').agg(s364=('sales_value','sum'))
    lastd = t.groupby('household_key')['day'].max()
    X = pd.DataFrame(index=hh)
    X = X.join(g28, how='left').join(g84, how='left').join(g364, how='left')
    for c, v in [('s28',0.0),('b28',0),('days28',0),('s84',0.0),('b84',0),('s364',0.0)]:
        X[c] = X[c].fillna(v)
    X['rec'] = (d - lastd).reindex(hh)
    X['rec'] = X['rec'].fillna(9999).astype(float)
    X['bv28'] = X.s28 / X.b28.replace(0, np.nan)
    X['trend'] = X.s28 / (X.s84 + 1.0)
    feats = {}
    for c in ['s28','s84','s364','b28','b84','days28','rec','bv28','trend']:
        feats['rk_'+c+'_pct'] = X[c].rank(pct=True)
    for c in ['s28','s84','b28','rec']:
        med = X[c].median(); iqr = X[c].quantile(0.75) - X[c].quantile(0.25)
        feats['rz_'+c] = ((X[c]-med)/(iqr+1e-9)).clip(-8, 8)
    feats['coh_median_s28'] = float(X.s28.median())
    feats['coh_mean_s28'] = float(X.s28.mean())
    feats['coh_iqr_s28'] = float(X.s28.quantile(0.75) - X.s28.quantile(0.25))
    feats['coh_active_rate'] = float((X.s28 > 0).mean())
    feats['coh_n'] = float(len(X))
    feats['coh_median_rec'] = float(X.rec.median())
    feats['coh_median_b28'] = float(X.b28.median())
    return pd.DataFrame(feats, index=hh)

df = agent_api.build_features(rank_block)
print('block built:', df.shape, df.columns.tolist())
e10 = agent_api.load_saved('e010_rhythm.parquet').drop(columns=['index'])
m = e10.merge(df, on=['household_key','snapshot_day'], how='inner')
print('merged:', m.shape)
p = agent_api.save_table(m, 'e011_rank.parquet')
print('saved:', p)
