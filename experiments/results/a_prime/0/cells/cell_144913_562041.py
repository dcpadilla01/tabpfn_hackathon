import numpy as np, pandas as pd

def structure_fn(view, snapshot_day):
    d = view.day
    keys = view.households.index if isinstance(view.households, pd.DataFrame) else pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.household_key.isin(keys)]
    out = pd.DataFrame(index=keys)

    t84 = tx[tx.day > d-84]
    g = t84.groupby('household_key')
    spend84 = g.sales_value.sum()
    out['s_gap_mean'] = (d - g.day.max())
    pdays = t84.groupby('household_key').day.apply(lambda s: np.sort(s.values))
    def gapstats(v):
        if len(v) < 2: return (np.nan, np.nan, np.nan, np.nan)
        gp = np.diff(v)
        return (gp.mean(), gp.std(), gp.max(), np.mean(gp>14))
    gs = pdays.apply(gapstats)
    out['gap_mean'] = gs.apply(lambda t: t[0])
    out['gap_std'] = gs.apply(lambda t: t[1])
    out['gap_max'] = gs.apply(lambda t: t[2])
    out['gap_gt14'] = gs.apply(lambda t: t[3])
    gm = out['gap_mean']
    out['gap_cur_ratio'] = out['s_gap_mean'] / gm.replace(0, np.nan)

    t12 = tx[tx.day > d-84]
    wk = ((t12.day + 8)//7)
    t12 = t12.assign(_w=wk)
    wact = t12.groupby('household_key')._w.nunique()
    out['wk_active_12'] = wact / 12.0
    wsp = t12.groupby(['household_key','_w']).sales_value.sum()
    ws = wsp.groupby(level=0)
    out['wk_cv'] = ws.std() / ws.mean().replace(0, np.nan)
    wcur = (d + 8)//7
    def streak(wsset):
        s = 0; w = wcur-1
        for k in sorted(wsset, reverse=True):
            if k == w: s += 1; w -= 1
            elif k < w: break
        return s
    out['streak_w'] = wact.index.to_series().map(lambda h: streak(set(wsp.loc[h].index)) if h in wsp.index else 0)

    st = t84.groupby(['household_key','store_id']).sales_value.sum()
    out['store_share_top'] = st.groupby(level=0).max() / spend84.replace(0, np.nan)
    out['n_stores84'] = t84.groupby('household_key').store_id.nunique()

    disc = (t84.retail_disc.fillna(0) + t84.coupon_disc.fillna(0) + t84.coupon_match_disc.fillna(0)).clip(lower=0)
    denom = (t84.sales_value + disc.clip(lower=0))
    out['disc_share'] = disc.groupby(t84.household_key).sum() / denom.groupby(t84.household_key).sum().replace(0, np.nan)

    upb = t84.groupby(['household_key','basket_id']).quantity.sum().groupby(level=0).mean()
    out['units_per_basket'] = upb

    tt = t84.trans_time.fillna(0)
    ev = (tt >= 1700).groupby(t84.household_key).mean()
    out['evening_share'] = ev

    pr = t84.groupby(['household_key','product_id']).sales_value.sum()
    out['prod_top_share'] = pr.groupby(level=0).max() / spend84.replace(0, np.nan)
    bk = t84.groupby(['household_key','basket_id']).sales_value.sum()
    out['basket_top_share'] = bk.groupby(level=0).max() / spend84.replace(0, np.nan)

    t28 = tx[tx.day > d-28]
    tprior = tx[(tx.day > d-112) & (tx.day <= d-28)]
    pr28 = t28.groupby(['household_key','product_id']).sales_value.sum()
    priorset = tprior.groupby('household_key').product_id.apply(set)
    def rep_share(h):
        if h not in pr28.index or h not in priorset.index: return np.nan
        s = pr28.loc[h]; ps = priorset.loc[h]
        if len(s)==0 or s.sum()==0: return np.nan
        m = pd.Series(s.index).isin(ps).values
        return float(s.values[m].sum() / s.sum())
    out['prod_repeat_28'] = pd.Series({h: rep_share(h) for h in keys})
    allprior = tx[tx.day <= d-28].groupby('household_key').product_id.apply(set)
    def new_share(h):
        if h not in pr28.index or h not in allprior.index: return np.nan
        s = pr28.loc[h]; ps = allprior.loc[h]
        if len(s)==0 or s.sum()==0: return np.nan
        m = pd.Series(s.index).isin(ps).values
        return float(s.values[~m].sum() / s.sum())
    out['prod_new_28'] = pd.Series({h: new_share(h) for h in keys})

    out['n_prod84'] = t84.groupby('household_key').product_id.nunique() / t84.groupby('household_key').basket_id.nunique().replace(0,np.nan)
    out['spend_per_active_day'] = spend84 / t84.groupby('household_key').day.nunique().replace(0,np.nan)
    return out

feats = build_features(structure_fn)
print("built:", feats.shape)
print("cols:", list(feats.columns))
print("NaN frac:")
print(feats.isna().mean().round(2).to_string())
save_table(feats.reset_index(), 'structure_v1')
print("saved")