import agent_api, pandas as pd, numpy as np

def fn(view, snapshot_day):
    tx = view.table('transactions')
    hh = view.households
    out = pd.DataFrame(index=hh)
    s = snapshot_day
    # weekly spend series, last 26 weeks
    w = tx[(tx.day > s - 26*7) & (tx.day <= s)].copy()
    w['wk'] = (w.day + 8)//7
    cur = (s + 8)//7
    weeks = np.arange(cur-25, cur+1)
    gws = w.groupby(['household_key','wk'])['sales_value'].sum()
    def wk_stats(t):
        v = t.reindex(pd.MultiIndex.from_product([t.index.get_level_values(0).unique(), weeks])).droplevel(0)
        v = v.groupby(level=0).sum() if False else v
        return v
    piv = gws.unstack('wk').reindex(columns=weeks, fill_value=0.0).reindex(hh).fillna(0.0)
    med = piv.median(axis=1).replace(0, np.nan)
    out['sk_wk_mean_over_med'] = (piv.mean(axis=1)/med).fillna(1.0).clip(0, 50)
    out['sk_wk_max_over_med'] = (piv.max(axis=1)/med).fillna(1.0).clip(0, 200)
    out['sk_wk_cv'] = (piv.std(axis=1)/piv.mean(axis=1).replace(0,np.nan)).fillna(0.0).clip(0, 10)
    out['sk_wk_zero_share'] = (piv == 0).mean(axis=1)
    # basket concentration, last 84d
    b = tx[(tx.day > s-84) & (tx.day <= s)]
    gb = b.groupby(['household_key','basket_id'])['sales_value'].sum()
    tot = gb.groupby('household_key').sum().reindex(hh).fillna(0.0)
    mx = gb.groupby('household_key').max().reindex(hh).fillna(0.0)
    medb = gb.groupby('household_key').median().reindex(hh)
    out['sk_basket_top_share'] = np.where(tot>0, mx/tot.replace(0,np.nan), 0.0).clip(0,1)
    out['sk_basket_mean_over_med'] = (gb.groupby('household_key').mean().reindex(hh)/medb).fillna(1.0).clip(0,50)
    out['sk_basket_n_per_wk'] = (gb.groupby('household_key').count().reindex(hh).fillna(0.0)/12.0)
    # aligned 4-week blocks, last 8 blocks: max/med, zero share
    blocks = np.zeros((len(hh), 8)); hh_idx = {h:i for i,h in enumerate(hh)}
    for k in range(8):
        hi = s - 28*k; lo = hi - 27
        sub = tx[(tx.day>=lo)&(tx.day<=hi)].groupby('household_key')['sales_value'].sum()
        for h, v in sub.items():
            blocks[hh_idx[h], k] = v
    bm = pd.DataFrame(blocks, index=hh)
    medb8 = bm.median(axis=1).replace(0, np.nan)
    out['sk_blk_max_over_med'] = (bm.max(axis=1)/medb8).fillna(1.0).clip(0,200)
    out['sk_blk_zero_share'] = (bm==0).mean(axis=1)
    out['sk_blk_med'] = bm.median(axis=1)          # historical median of 4w-block spend
    out['sk_blk_mean'] = bm.mean(axis=1)
    out['sk_p_zero_next'] = (bm==0).mean(axis=1)   # same as zero share (historical P(block=0))
    # interaction candidates: median level x zero risk
    out['sk_med_x_zerorisk'] = out['sk_blk_med'] * out['sk_blk_zero_share']
    out['sk_med_x_wkzeros'] = out['sk_blk_med'] * out['sk_wk_zero_share']
    return out

sk = agent_api.build_features(fn)
print('skew feats', sk.shape)
agent_api.save_table(sk, 'cand_skew')
print(sk.describe().T[['mean','50%','max']].round(3))
