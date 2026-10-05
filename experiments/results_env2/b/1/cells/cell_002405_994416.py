import numpy as np, pandas as pd, agent_api, warnings
warnings.filterwarnings('ignore')

def cand_fn(view, snapshot_day):
    hh = view.households
    tx = view.table('transactions')
    d0 = view.day
    tx = tx[tx.household_key.isin(set(hh))]
    tx = tx.assign(bs=tx.groupby('basket_id').sales_value.transform('sum'))
    out = pd.DataFrame(index=hh)
    def win(days): return tx[(tx.day > d0-days) & (tx.day <= d0)]
    w84 = win(84)
    g = w84.groupby('household_key')
    tot = g.sales_value.sum()
    # A: tail / premium
    bs84 = w84.drop_duplicates('basket_id')[['household_key','bs']]
    gb = bs84.groupby('household_key').bs
    out['n_big84'] = gb.apply(lambda s: (s>100).sum())
    out['bigshare84'] = gb.apply(lambda s: s[s>100].sum()).fillna(0)/(tot+1e-9)
    out['top3b84'] = gb.apply(lambda s: s.nlargest(3).sum() if len(s)>=1 else 0)/(tot+1e-9)
    out['maxline84'] = w84.groupby('household_key').sales_value.max()
    hi = w84[w84.sales_value>=15]
    out['hi_item_share84'] = hi.groupby('household_key').sales_value.sum().fillna(0)/(tot+1e-9)
    # B: absolute department spend 84d (merge products)
    pr = view.table('products')[['product_id','department']]
    wdep = w84.merge(pr, on='product_id', how='left')
    dep = wdep.groupby(['household_key','department']).sales_value.sum().unstack(fill_value=0.0)
    for dd in ['GROCERY','MEAT','PRODU','DELI','SPIRI','COSME','FLORA','SEAFO']:
        out['dabs84_'+dd] = dep[dd] if dd in dep.columns else 0.0
    # D: dormancy
    days = tx[tx.day > d0-364].sort_values(['household_key','day'])
    gd = days.groupby('household_key').day.diff()
    gap_med = gd.groupby(days.household_key).median()
    dsl = (d0 - g.day.max()).clip(lower=0)
    out['dorm_r'] = dsl/(gap_med+1.0)
    out['act_exp'] = np.exp(-dsl/(gap_med+7.0))
    z = 0.0
    for k in range(3):
        wk = win(28*(k+1)).groupby('household_key').sales_value.sum()
        z = z + (wk.reindex(hh).fillna(0)==0).astype(float)
    out['zeros_last3'] = z
    # E: discount intensity
    out['cd84_r'] = w84.groupby('household_key').coupon_disc.sum().abs()/(tot+1e-9)
    out['rd84_r'] = w84.groupby('household_key').retail_disc.sum().abs()/(tot+1e-9)
    # F: time of day
    out['eve_share84'] = w84[w84.trans_time>=1700].groupby('household_key').sales_value.sum().fillna(0)/(tot+1e-9)
    out['morn_share84'] = w84[w84.trans_time<1000].groupby('household_key').sales_value.sum().fillna(0)/(tot+1e-9)
    # H: unit price
    q = w84.groupby('household_key').quantity.sum().clip(lower=0.1)
    out['up84'] = tot/q
    # C: cross-sectional ranks
    out['r_sp84'] = tot.reindex(hh).fillna(0).rank(pct=True)
    out['r_dsl'] = dsl.reindex(hh).fillna(999).rank(pct=True)
    tr84 = w84.groupby('household_key').basket_id.nunique().reindex(hh).fillna(0)
    out['r_tr84'] = tr84.rank(pct=True)
    return out

X = agent_api.build_features(cand_fn)
print(X.shape)
tt = agent_api.train_targets()
m = tt.merge(X.reset_index(), on=['household_key','snapshot_day'])
agent_api.save_table(X.reset_index(), 'cand_new1.parquet')
print('saved', m.shape)
