import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

def _idx(h):
    if isinstance(h, pd.DataFrame): return h.index
    return pd.Index(h)

# ---- probe: can fn() use load_saved? ----
def probe(view, sd):
    ok = 0.0
    try:
        e = agent_api.load_saved('e012_style.parquet')
        e = e[e.snapshot_day == sd]
        ok = 2.0 if len(e) > 0 else 0.5
    except Exception:
        ok = 0.0
    return pd.DataFrame({'ok': [ok]}, index=_idx(view.households))

res = agent_api.build_features(probe)
vals = sorted(res.ok.value_counts().to_dict().items())
print("PROBE:", vals)
LOAD_OK = (len(vals) == 1 and vals[0][0] == 2.0)
print("LOAD_OK:", LOAD_OK)

def fn(view, sd):
    idx = _idx(view.households)
    tx = view.table('transactions')
    tx = tx[tx.household_key.isin(set(idx))].copy()
    t = float(sd)
    tx['age'] = t - tx.day.values.astype(float)
    hh = tx.household_key.values
    sv = tx.sales_value.values.astype(float)
    age = tx.age.values
    out = pd.DataFrame(index=idx)

    def agg(mask, val):
        s = pd.Series(np.asarray(val, dtype=float)[mask], index=hh[mask])
        return s.groupby(level=0).sum().reindex(idx).fillna(0.0)

    for k in range(1, 9):  # weekly spend buckets, most recent first
        out[f'w{k}'] = agg((age < 7*k) & (age >= 7*(k-1)), sv)
    out['spend_168'] = agg(age < 168, sv)
    for hl in [7.0, 21.0, 42.0, 84.0]:  # finer EWMA grid
        out[f'e_hl{int(hl)}'] = agg(age <= 365, sv * 0.5**(age/hl))
    # EWMA trips (per basket), hl 28
    b = tx[age <= 365].groupby(['household_key','basket_id']).agg(bday=('day','first'))
    bd = t - b.bday.values.astype(float)
    out['e_trips_hl28'] = pd.Series(0.5**(bd/28.0), index=b.index.get_level_values(0))\
        .groupby(level=0).sum().reindex(idx).fillna(0.0)
    # weekly series stats (12 weeks)
    wk = (age // 7).astype(int)
    m12 = wk < 12
    dfw = pd.DataFrame({'hh': hh[m12], 'wk': wk[m12], 'v': sv[m12]})
    W = dfw.pivot_table(index='hh', columns='wk', values='v', aggfunc='sum')\
        .reindex(columns=range(12)).reindex(idx).fillna(0.0).values
    mn = W.mean(1); s_ = W.std(1)
    out['wk_mean12'] = mn
    out['wk_cv12'] = np.where(mn > 1e-9, s_/np.maximum(mn, 1e-9), 0.0)
    a, b2 = W[:, :-1], W[:, 1:]
    am = a - a.mean(1, keepdims=True); bm = b2 - b2.mean(1, keepdims=True)
    den = np.sqrt((am**2).sum(1)*(bm**2).sum(1))
    out['wk_ac1'] = np.where(den > 1e-9, (am*bm).sum(1)/np.maximum(den, 1e-9), 0.0)
    # big-trip concentration (84d)
    m84 = age < 84
    bb = tx[m84].groupby(['household_key','basket_id']).sales_value.sum()
    g = bb.groupby(level=0)
    mx = g.max().reindex(idx).fillna(0.0)
    tot = g.sum().reindex(idx).fillna(0.0)
    out['bt_max_84'] = mx
    out['bt_cnt100_84'] = g.apply(lambda s: float((s > 100).sum())).reindex(idx).fillna(0.0)
    out['bt_top1_share_84'] = np.where(tot > 0, mx/np.where(tot > 0, tot, 1.0), np.nan)

    if LOAD_OK:
        base = agent_api.load_saved('e012_style.parquet')
        base = base[base.snapshot_day == sd].set_index('household_key').drop(columns=['snapshot_day'])
        return out.join(base, how='left')
    return out

tab = agent_api.build_features(fn)
print("built:", tab.shape)
print("has e012 cols:", 'spend_84' in tab.columns, "| has new:", 'e_hl84' in tab.columns, 'wk_ac1' in tab.columns)
p = agent_api.save_table(tab, 'e014_recency.parquet')
print("saved:", p)
