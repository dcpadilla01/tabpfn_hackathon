import agent_api as A, pandas as pd, numpy as np

base = A.load_saved("churn_seasonality.parquet")

def weekly_feats(view, snapshot_day):
    t = view.transactions
    hh = view.households
    idx = hh.index if isinstance(hh, pd.DataFrame) else pd.Index(hh, name='household_key')
    cur_week = (snapshot_day + 8) // 7
    all_weeks = list(range(1, cur_week))
    W = len(all_weeks)

    wsp = t.groupby(['household_key','week_no'])['sales_value'].sum()
    tb = t.drop_duplicates(subset=['household_key','basket_id'])
    wtr = tb.groupby(['household_key','week_no']).size()
    wln = t.groupby(['household_key','week_no']).size()

    Psp = wsp.unstack('week_no').reindex(index=idx, columns=all_weeks).fillna(0.0)
    Ptr = wtr.unstack('week_no').reindex(index=idx, columns=all_weeks).fillna(0.0)
    Pln = wln.unstack('week_no').reindex(index=idx, columns=all_weeks).fillna(0.0)

    out = pd.DataFrame(index=idx)
    sp = Psp.values; tr = Ptr.values; ln = Pln.values
    for k in range(1, 27):
        out[f'ws_{k}'] = sp[:, W-k] if W-k >= 0 else 0.0
    for k in range(1, 14):
        out[f'wt_{k}'] = tr[:, W-k] if W-k >= 0 else 0.0
        out[f'wl_{k}'] = ln[:, W-k] if W-k >= 0 else 0.0
    for j in range(1, 7):
        lo = max(W - 4*j, 0); hi = W - 4*(j-1)
        out[f'wblk_{j}'] = sp[:, lo:hi].sum(axis=1)
    yoy = sp[:, max(W-56,0):max(W-52,0)].sum(axis=1)
    out['wyoy'] = yoy
    out['wyoy_ratio'] = yoy / (out['wblk_1'] + 5.0)
    lo26 = max(W-26, 0)
    out['wmean26'] = sp[:, lo26:W].mean(axis=1)
    out['wmax26'] = sp[:, lo26:W].max(axis=1)
    out['wstd26'] = sp[:, lo26:W].std(axis=1)
    out['wnz26'] = (sp[:, lo26:W] > 0).sum(axis=1)
    out['wshare_last4'] = out['wblk_1'] / (out['wmean26']*4 + 1e-6)
    n26 = min(W, 26)
    if n26 >= 2:
        y = sp[:, W-n26:W]; x = np.arange(n26, dtype=float); xc = x - x.mean()
        out['wslope26'] = (y * xc).sum(axis=1) / (xc*xc).sum()
    else:
        out['wslope26'] = 0.0
    act = sp[:, lo26:W] > 0
    rev = act[:, ::-1]
    has = rev.any(axis=1)
    out['wks_since_active'] = np.where(has, rev.argmax(axis=1), n26)
    out['wt_mean13'] = tr[:, max(W-13,0):W].mean(axis=1)
    out['wl_mean13'] = ln[:, max(W-13,0):W].mean(axis=1)
    return out

wk = A.build_features(weekly_feats)
print(wk.shape)
print(wk.head(3).T.head(15))
full = base.merge(wk.reset_index(), on=['household_key','snapshot_day'], how='inner')
print("merged:", full.shape, "base:", base.shape)
p = A.save_table(full, "weekly_history")
print(p)