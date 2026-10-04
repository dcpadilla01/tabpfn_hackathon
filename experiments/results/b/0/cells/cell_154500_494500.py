e = load_saved('e011_price.parquet')
t = train_targets()
keys = ['household_key','snapshot_day']
e = e.sort_values(keys).reset_index(drop=True)

# fixed, data-independent bucket cuts
def sp_b(s):
    s = s.fillna(-1.0)
    return np.digitize(s, [0.01, 20, 50, 100, 175, 300])  # 0..6 (NaN/-1 -> 0)
def rec_b(s):
    s = s.fillna(999.0)
    return np.digitize(s, [8, 15, 29, 57])                # 0..4
def tr_b(s):
    s = s.fillna(0.0)
    return np.digitize(s, [1, 4, 7, 11])                  # 0..4
def sp_coarse(s):
    s = s.fillna(-1.0)
    return np.digitize(s, [0.01, 50, 150])                # 0..3
def rec_coarse(s):
    s = s.fillna(999.0)
    return np.digitize(s, [8, 29])                        # 0..2

e['sp_b'] = sp_b(e['spend28']); e['rec_b'] = rec_b(e['recency'])
e['tr_b'] = tr_b(e['trips28']); e['lg_b'] = sp_b(e['lag_spend_1'])
e['cross_b'] = sp_coarse(e['spend28'])*10 + rec_coarse(e['recency'])
bcols = ['sp_b','rec_b','tr_b','lg_b','cross_b']
tt = t.merge(e[keys+bcols], on=keys, how='left')
print('tt rows', len(tt), 'nan buckets', int(tt['sp_b'].isna().sum()))

K_MEAN, K_ZERO = 30.0, 50.0
te_cols = ['te_sp28','te_sp28z','te_rec','te_trip','te_lag1','te_cross','prior_mean']
TE = np.full((len(e), len(te_cols)), np.nan)
fam = [('sp_b','te_sp28','te_sp28z'), ('rec_b','te_rec',None), ('tr_b','te_trip',None),
       ('lg_b','te_lag1',None), ('cross_b','te_cross',None)]
ycol = TARGET
for d in sorted(e['snapshot_day'].unique()):
    idx = (e['snapshot_day']==d).values
    hist = tt[tt['snapshot_day'] < d]
    if len(hist)==0:
        continue
    prior = float(hist[ycol].mean()); pz = float((hist[ycol]==0).mean())
    TE[idx, te_cols.index('prior_mean')] = prior
    for bcol, mcol, zcol in fam:
        g = hist.groupby(bcol)[ycol]
        n = g.size(); s = g.sum(); z = g.apply(lambda v: (v==0).mean())
        gm = (s + K_MEAN*prior)/(n + K_MEAN)
        rb = e.loc[idx, bcol]
        TE[idx, te_cols.index(mcol)] = rb.map(gm).values
        if zcol:
            gz = (g.apply(lambda v:(v==0).sum()) + K_ZERO*pz)/(n + K_ZERO)
            TE[idx, te_cols.index(zcol)] = rb.map(gz).values
te_df = pd.DataFrame(TE, columns=te_cols)
out = pd.concat([e, te_df], axis=1)
# leakage sanity checks
m = out.merge(t, on=keys, how='inner')
tr = m[m.snapshot_day>=123]
for c in te_cols:
    cc = np.corrcoef(tr[c].fillna(tr[c].mean()), tr[ycol])[0,1]
    print(f'{c:10s} corr_with_y(train)={cc:6.3f}  nan_frac={out[c].isna().mean():.3f}')
print('snapshot 95 te non-nan:', int(out[out.snapshot_day==95][te_cols].notna().any(axis=1).sum()))
print('rows', len(out), 'cols', out.shape[1])