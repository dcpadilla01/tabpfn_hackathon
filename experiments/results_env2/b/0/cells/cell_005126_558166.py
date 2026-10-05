import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    tx = view.table('transactions')
    hh = view.households
    g = tx.groupby('household_key')
    out = pd.DataFrame(index=hh)
    # aligned 28d blocks b1..b5: [s-27..s], [s-55..s-28], ...
    for k in range(1, 6):
        hi = snapshot_day - 28*(k-1); lo = hi - 27
        out['b%d' % k] = g.apply(lambda t, lo=lo, hi=hi: t.loc[(t.day>=lo)&(t.day<=hi),'sales_value'].sum()) 
    w = tx.copy(); w['wk'] = (w.day + 8)//7
    cur_wk = (snapshot_day + 8)//7
    gw = w.groupby(['household_key','wk'])['sales_value'].sum().reset_index()
    for nwk, tag in [(12,'12'), (26,'26')]:
        sub = gw[(gw.wk <= cur_wk) & (gw.wk > cur_wk - nwk)]
        full = pd.DataFrame(index=hh); full['s'] = 0.0
        piv = sub.groupby('household_key')['sales_value'].sum()
        # include zero weeks: reindex over all weeks in window
        def med(t, nwk=nwk):
            weeks = np.arange(cur_wk-nwk+1, cur_wk+1)
            s = t.set_index('wk')['sales_value'].reindex(weeks).fillna(0.0).values
            return np.median(s)
        out['med_week_%s' % tag] = hh.map(sub.groupby('household_key').apply(med)).fillna(0.0)
    # median basket value last 84d
    sub84 = tx[tx.day > snapshot_day-84]
    out['med_basket_84'] = hh.map(sub84.groupby(['household_key','basket_id'])['sales_value'].sum().groupby('household_key').median())
    # robust block combos
    b = out[['b1','b2','b3','b4']].values
    out['blk_med3'] = np.median(b[:, :3], axis=1)
    out['blk_med4'] = np.median(b, axis=1)
    out['blk_min12'] = np.minimum(b[:,0], b[:,1])
    out['blk_min123'] = np.min(b[:, :3], axis=1)
    out['blk_trim4'] = (np.sort(b, axis=1)[:,1] + np.sort(b, axis=1)[:,2]) / 2.0
    out['blk_wavg'] = 0.4*b[:,0] + 0.3*b[:,1] + 0.2*b[:,2] + 0.1*b[:,3]
    out['blk_iqr4'] = np.sort(b, axis=1)[:,2] - np.sort(b, axis=1)[:,1]
    out['blk_zero_share4'] = (b == 0).mean(axis=1)
    out = out.drop(columns=['b1','b2','b3','b4'])
    return out

cand = agent_api.build_features(fn)
print('cand', cand.shape)
agent_api.save_table(cand, 'cand_robust')
print(cand.head())
