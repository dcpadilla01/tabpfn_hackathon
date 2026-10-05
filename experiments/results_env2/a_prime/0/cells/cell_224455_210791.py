import agent_api, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

# Candidate: recency-weighted (exponential decay) spend/basket features
def fn(view, snapshot_day):
    tx = view.table('transactions')
    d = tx['day'].values.astype(float)
    sv = tx['sales_value'].values.astype(float)
    hh = tx['household_key'].values
    bid = tx['basket_id'].values
    out = pd.DataFrame(index=pd.Index(sorted(tx['household_key'].unique()), name='household_key'))
    for hl in (14, 28, 56, 112):
        lam = np.log(2.0)/hl
        w = np.exp(-lam*(snapshot_day - d))
        df = pd.DataFrame({'hh': hh, 'sw': sv*w, 'wl': np.log1p(np.maximum(sv,0))*w, 'w': w})
        agg = df.groupby('hh').agg(ew_spend=('sw','sum'), ew_log=('wl','sum'), wmass=('w','sum'))
        out['ew_spend_hl%d'%hl] = agg['ew_spend']
        out['ew_rate_hl%d'%hl] = agg['ew_spend']*lam
        out['ew_log_hl%d'%hl] = agg['ew_log']/agg['wmass']
        # basket-level weighted count
        b = pd.DataFrame({'hh': hh, 'bid': bid, 'day': d}).drop_duplicates('bid')
        bw = np.exp(-lam*(snapshot_day - b['day'].values))
        out['ew_bask_hl%d'%hl] = pd.Series(bw, index=b['hh']).groupby(level=0).sum()
    return out

cand = agent_api.build_features(fn)
print('cand', cand.shape)

base = agent_api.load_saved('e007_te.parquet')
tt = agent_api.train_targets()
d = base.merge(cand.reset_index(), on=['household_key','snapshot_day']).merge(tt, on=['household_key','snapshot_day'])
print('merged', d.shape)
agent_api.save_table(d.drop(columns=['future_spend_4w']), 'e009_ewma.parquet')
print('saved e009_ewma.parquet')
