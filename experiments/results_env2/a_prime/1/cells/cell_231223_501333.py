import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

e6 = agent_api.load_saved('e006_dynamics.parquet')

def fn(view, snapshot_day):
    tx = view.table('transactions')
    wk = int(view.week)
    lo = max(1, wk-77)
    tx = tx[(tx.week_no >= lo) & (tx.week_no <= wk)]
    g = tx.groupby(['household_key','week_no']).sales_value.sum()
    weeks = np.arange(lo, wk+1)
    ages = wk - weeks
    w28 = 0.5**(ages/4.0); w56 = 0.5**(ages/8.0)
    out = {}
    for h, sub in g.groupby(level=0):
        s = sub.droplevel(0)
        a = pd.Series(0.0, index=weeks)
        a.loc[s.index] = (s > 0).astype(float).values
        out[h] = dict(
            act_ewma28=float((a*w28).sum()/w28.sum()),
            act_ewma56=float((a*w56).sum()/w56.sum()),
            act_share_13w=float(a.iloc[-13:].mean()),
            act_share_26w=float(a.iloc[-26:].mean()),
            act_share_52w=float(a.iloc[-52:].mean()),
        )
    df = pd.DataFrame.from_dict(out, orient='index')
    df.index.name='household_key'
    return df.reindex(view.households)

probe = agent_api.build_features(fn)
print('probe shape', probe.shape)
agent_api.save_table(probe, 'e011_probe.parquet')

# assemble candidates
p = probe.copy()
base = e6.merge(p.reset_index(), on=['household_key','snapshot_day'], how='inner')
th=5.0
base['sp28_lo']=np.minimum(base.spend_28,th); base['sp28_hi']=np.maximum(base.spend_28-th,0)
base['ew28_lo']=np.minimum(base.d_ewma_spend_hl28,th); base['ew28_hi']=np.maximum(base.d_ewma_spend_hl28-th,0)
base['ew28_lo25']=np.minimum(base.d_ewma_spend_hl28,25.0); base['ew28_hi25']=np.maximum(base.d_ewma_spend_hl28-25.0,0)
base['lapsed']=(base.spend_28<th).astype(float)
base['exp_spend']=base.spend_28*base.act_ewma28
base['exp_spend2']=base.d_ewma_spend_hl28*base.act_ewma28
base['rec_lo']=np.minimum(base.recency,28.0); base['rec_hi']=np.maximum(base.recency-28.0,0)
agent_api.save_table(base, 'e011_cand.parquet')
t = agent_api.train_targets()
d = base.merge(t, on=['household_key','snapshot_day'])
print('cand rows', len(d))
print(d[['act_ewma28','act_share_26w','sp28_lo','lapsed','exp_spend']].describe().round(3))
