def fn(view, snapshot_day):
    hh = view.households
    tx = view.transactions
    H = {}
    for h in hh:
        H[h] = agent_api.history(h)
    rows = {}
    for h in hh:
        df = H[h]
        d = df['day'].values; s = df['sales_value'].values
        tot = s.sum()
        w28 = []
        for k in range(1, 14):  # 13 trailing 28d windows
            lo, hi = snapshot_day-28*k, snapshot_day-28*(k-1)
            m = (d > lo) & (d <= hi)
            w28.append(s[m].sum())
        w28 = np.array(w28)
        wk = []
        for k in range(1, 9):  # last 8 full weeks
            lo, hi = snapshot_day-7*k, snapshot_day-7*(k-1)
            m = (d > lo) & (d <= hi)
            wk.append(s[m].sum())
        wk = np.array(wk)
        act = d[d > snapshot_day-728]
        r = {
            'spend_728': s[d > snapshot_day-728].sum(),
            'spend_546': s[d > snapshot_day-546].sum(),
            'spend_112': s[d > snapshot_day-112].sum(),
            'spend_140': s[d > snapshot_day-140].sum(),
            'spend_168': s[d > snapshot_day-168].sum(),
            'mean28_13w': w28.mean(),
            'std28_13w': w28.std(),
            'max28_13w': w28.max(),
            'min28_13w': w28.min(),
            'cv28': w28.std()/(w28.mean()+1e-6),
            'mean_wk8': wk.mean(),
            'std_wk8': wk.std(),
            'cv_wk8': wk.std()/(wk.mean()+1e-6),
            'max_wk8': wk.max(),
            'wk_rate_all': tot/max((snapshot_day - d.min())/7.0, 1e-6) if len(d) else 0.0,
            'n_active_days_728': len(np.unique(act)) if len(act) else 0,
            'nb_728': len(act),
            'avg_basket_728': s[d > snapshot_day-728].sum()/max(len(act),1) if len(act) else 0.0,
            'spend_28_minus_336': s[(d > snapshot_day-364) & (d <= snapshot_day-336)].sum(),
            'spend_28_minus_672': s[(d > snapshot_day-700) & (d <= snapshot_day-672)].sum(),
            'spend_28_minus_504': s[(d > snapshot_day-532) & (d <= snapshot_day-504)].sum(),
        }
        rows[h] = r
    return pd.DataFrame(rows).T

X = agent_api.build_features(fn)
base = agent_api.load_saved('e001_history.parquet')
feat = base.merge(X, on=['household_key','snapshot_day'], suffixes=('','_n'))
feat = feat.drop(columns=[c for c in feat.columns if c.endswith('_n')])
p = agent_api.save_table(feat, 'e004_long_hist.parquet')
print(p, feat.shape)