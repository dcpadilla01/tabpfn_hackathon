import agent_api, pandas as pd
t = agent_api.load_saved('e009_ewma_longlags.parquet')
cols = list(t.columns)
print(len(cols))
print(cols)


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def make_feats(view, D):
    H = pd.Index(view.households)
    tx = view.transactions
    tx = tx[tx.day >= D-363] if D-363 > 0 else tx
    g = tx.groupby('household_key')

    out = pd.DataFrame(index=H)

    # ---- store affinity (84d) ----
    t84 = tx[tx.day >= D-83]
    st = t84.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    tot = st.groupby('household_key').sales_value.sum().rename('tot')
    st = st.join(tot, on='household_key')
    st['sh'] = st.sales_value / st.tot.replace(0, np.nan)
    st = st.sort_values(['household_key','sh'], ascending=[True, False])
    def topk(g, k):
        vals = g.sh.values
        return pd.Series({'s%d'%k: vals[k-1] if len(vals) >= k else np.nan}, index=['s%d'%k])
    tops = st.groupby('household_key').apply(lambda g: pd.Series({
        'store_top1_share84': g.sh.iloc[0] if len(g) else np.nan,
        'store_top2_share84': g.sh.iloc[:2].sum() if len(g) >= 2 else g.sh.sum(),
        'store_top3_share84': g.sh.iloc[:3].sum() if len(g) >= 3 else g.sh.sum(),
        'store_hhi84': float((g.sh.fillna(0)**2).sum()),
        'store_entropy84': float(-(g.sh.fillna(0)*(g.sh.fillna(0).clip(lower=1e-9)).apply(np.log)).sum()),
        'store_n84': float(len(g)),
    }))
    out = out.join(tops)

    # ---- basket shape / stock-up vs top-up (84d) ----
    b = t84.groupby(['household_key','basket_id']).agg(
        sv=('sales_value','sum'), ln=('sales_value','size'), qty=('quantity','sum'),
        day=('day','first'))
    b = b.reset_index()
    bg = b.groupby('household_key')
    out['basket_max_84b'] = bg.sv.max()
    out['basket_min_84b'] = bg.sv.min()
    out['basket_med_84b'] = bg.sv.median()
    out['basket_p90_84'] = bg.sv.quantile(0.9)
    out['stockup_share84'] = bg.apply(lambda g: (g.sv > 1.5*max(g.sv.median(), 1e-9)).mean())
    out['lines_max_84'] = bg.ln.max()
    out['qty_max_84'] = bg.qty.max()
    out['bigtrip_spend_share84'] = bg.apply(lambda g: g.sv[g.sv > 1.5*max(g.sv.median(),1e-9)].sum()/max(g.sv.sum(),1e-9))

    # ---- lifetime zero-streak / churn history (up to 363d) ----
    wk = tx.copy()
    wk['week_no'] = ((wk.day + 8) // 7).astype(int)
    ww = wk.groupby(['household_key','week_no']).sales_value.sum().reset_index()
    ww = ww[ww.week_no >= ((D-363)+8)//7]
    def streaks(g):
        s = (g.sales_value > 0).astype(int).values
        # current trailing zero streak
        z = 0
        for v in s[::-1]:
            if v == 0: z += 1
            else: break
        # longest zero streak
        best = cur = 0
        for v in s:
            if v == 0:
                cur += 1; best = max(best, cur)
            else: cur = 0
        return pd.Series({'zero_streak_now': z, 'zero_streak_max': best,
                          'active_weeks': s.sum(), 'n_weeks': len(s)})
    zs = ww.groupby('household_key').apply(streaks)
    out = out.join(zs)
    out['active_share_52w'] = out.active_weeks / out.n_weeks.replace(0, np.nan)

    # ---- discount-type mix (84d) ----
    d84 = t84.groupby('household_key').agg(
        retail=('retail_disc','sum'), coup=('coupon_disc','sum'),
        cmatch=('coupon_match_disc','sum'), sv=('sales_value','sum'))
    out['retail_disc_share84'] = d84.retail.abs()/d84.sv.replace(0,np.nan)
    out['coup_disc_share84'] = d84.coup.abs()/d84.sv.replace(0,np.nan)
    out['cmatch_share84'] = d84.cmatch.abs()/d84.sv.replace(0,np.nan)

    # ---- demographic x spend interactions ----
    dem = view.demographics.set_index('household_key')
    out['spend28_x_size'] = out.spend28_dummy if False else np.nan
    return out, tx

# need spend_28 etc from saved e009 table for interactions
base = agent_api.load_saved('e009_ewma_longlags.parquet')
print(base.shape)


# ---- cell ----
import agent_api, numpy as np, pandas as pd

def make_feats(view, D):
    H = pd.Index(view.households)
    tx = view.transactions
    if D-363 > 0:
        tx = tx[tx.day >= D-363]
    out = pd.DataFrame(index=H)

    # ---- store affinity (84d) ----
    t84 = tx[tx.day >= D-83]
    st = t84.groupby(['household_key','store_id']).sales_value.sum().reset_index()
    tot = st.groupby('household_key').sales_value.sum().rename('tot')
    st = st.join(tot, on='household_key')
    st['sh'] = (st.sales_value / st.tot.replace(0, np.nan)).fillna(0)
    tops = st.groupby('household_key').apply(lambda g: pd.Series({
        'store_top1_share84': g.sh.iloc[0],
        'store_top2_share84': g.sh.iloc[:2].sum(),
        'store_top3_share84': g.sh.iloc[:3].sum(),
        'store_hhi84': float((g.sh**2).sum()),
        'store_entropy84': float(-(g.sh[g.sh>0]*np.log(g.sh[g.sh>0])).sum()),
        'store_n84': float(len(g)),
    }))
    out = out.join(tops)

    # ---- basket shape / stock-up vs top-up (84d) ----
    b = t84.groupby(['household_key','basket_id']).agg(
        sv=('sales_value','sum'), ln=('sales_value','size'), qty=('quantity','sum')).reset_index()
    bg = b.groupby('household_key')
    out['basket_min_84b'] = bg.sv.min()
    out['basket_med_84b'] = bg.sv.median()
    out['basket_p90_84'] = bg.sv.quantile(0.9)
    out['lines_max_84'] = bg.ln.max()
    out['qty_max_84'] = bg.qty.max()
    med = bg.sv.median().rename('med')
    b = b.join(med, on='household_key')
    b['is_stockup'] = b.sv > 1.5*b.med.replace(0, np.nan)
    gsum = b.groupby('household_key').agg(stockup_share=('is_stockup','mean'),
                                          bigtrip_share=('sv', lambda s: np.nan))
    b['big'] = b.sv.where(b.is_stockup, 0.0)
    bt = (b.groupby('household_key').big.sum() / b.groupby('household_key').sv.sum().replace(0,np.nan)).rename('bigtrip_share84')
    out['stockup_share84'] = gsum.stockup_share
    out['bigtrip_share84'] = bt

    # ---- weekly zero-streak / churn history (52w) ----
    wk = tx.copy()
    wk['week_no'] = ((wk.day + 8) // 7).astype(int)
    wmin = ((max(D-363,1)) + 8) // 7
    ww = wk[wk.week_no >= wmin].groupby(['household_key','week_no']).sales_value.sum().reset_index()
    def streaks(g):
        s = (g.sales_value > 0).astype(int).values
        z = 0
        for v in s[::-1]:
            if v == 0: z += 1
            else: break
        best = cur = 0
        for v in s:
            if v == 0:
                cur += 1; best = max(best, cur)
            else: cur = 0
        return pd.Series({'zero_streak_now': z, 'zero_streak_max': best,
                          'active_weeks': s.sum(), 'n_weeks': len(s)})
    zs = ww.groupby('household_key').apply(streaks)
    out = out.join(zs)
    out['active_share_52w'] = out.active_weeks / out.n_weeks.replace(0, np.nan)

    # ---- discount-type mix (84d) ----
    d84 = t84.groupby('household_key').agg(
        retail=('retail_disc','sum'), coup=('coupon_disc','sum'),
        cmatch=('coupon_match_disc','sum'), sv=('sales_value','sum'))
    out['retail_disc_share84'] = d84.retail.abs()/d84.sv.replace(0,np.nan)
    out['coup_disc_share84'] = d84.coup.abs()/d84.sv.replace(0,np.nan)
    out['cmatch_share84'] = d84.cmatch.abs()/d84.sv.replace(0,np.nan)
    return out

new = agent_api.build_features(make_feats)
print(new.shape)
print(new.head())
agent_api.save_table(new, 'e013_new_feats.parquet')


# ---- cell ----
import agent_api, numpy as np, pandas as pd

base = agent_api.load_saved('e009_ewma_longlags.parquet')
new = agent_api.load_saved('e013_new_feats.parquet')

m = base.merge(new, on=['household_key','snapshot_day'], how='left')
print(m.shape)

# demographic ordinal mapping + interactions
dem = agent_api.snapshot().demographics.set_index('household_key')
def to_num(s, prefix=None):
    if prefix:
        return s.str.extract(r'(\d+)').astype(float)[0]
    return pd.to_numeric(s, errors='coerce')

age = dem.classification_1.str.extract(r'(\d+)').astype(float)[0]
size = dem.classification_4.str.extract(r'(\d+)').astype(float)[0]
lvl = dem.classification_3.str.extract(r'(\d+)').astype(float)[0]
grp = dem.classification_5.str.extract(r'(\d+)').astype(float)[0]
kid = dem.kid_category_desc.map({'None/Unknown':0,'1':1,'2':2,'3+':3})
home = dem.homeowner_desc.map({'Renter':0,'Probable Renter':1,'Unknown':2,'Probable Owner':3,'Homeowner':4})
dnum = pd.DataFrame({'age_n':age,'size_n':size,'lvl_n':lvl,'grp_n':grp,'kid_n':kid,'home_n':home})
dnum.index.name = 'household_key'

m = m.merge(dnum.reset_index(), on='household_key', how='left')
m['ix_spend28_size'] = m.spend_28 * m.size_n
m['ix_spend28_kid'] = m.spend_28 * m.kid_n
m['ix_spend28_age'] = m.spend_28 * m.age_n
m['ix_ewma4_size'] = m.ewma_4 * m.size_n
m['ix_ewma4_kid'] = m.ewma_4 * m.kid_n
m['ix_tlag_size'] = m.tlag_mean * m.size_n
m['ix_spend28_home'] = m.spend_28 * m.home_n
m['ix_size_ewma8'] = m.ewma_8 * m.size_n
print(m.shape)
agent_api.save_table(m, 'e013_full.parquet')


# ---- cell ----
import agent_api, numpy as np, pandas as pd

base = agent_api.load_saved('e009_ewma_longlags.parquet')
new = agent_api.load_saved('e013_new_feats.parquet')

m = base.merge(new, on=['household_key', 'snapshot_day'], how='left')

dem = agent_api.snapshot().demographics.set_index('household_key')
age = dem.classification_1.str.extract(r'(\d+)').astype(float)[0]
size = dem.classification_4.str.extract(r'(\d+)').astype(float)[0]
lvl = dem.classification_3.str.extract(r'(\d+)').astype(float)[0]
grp = dem.classification_5.str.extract(r'(\d+)').astype(float)[0]
kid = dem.kid_category_desc.map({'None/Unknown': 0, '1': 1, '2': 2, '3+': 3})
home = dem.homeowner_desc.map({'Renter': 0, 'Probable Renter': 1, 'Unknown': 2, 'Probable Owner': 3, 'Homeowner': 4})
dnum = pd.DataFrame({'age_n': age, 'size_n': size, 'lvl_n': lvl, 'grp_n': grp, 'kid_n': kid, 'home_n': home})
dnum.index.name = 'household_key'

m = m.merge(dnum.reset_index(), on='household_key', how='left')
for c in ['size_n', 'kid_n', 'age_n', 'home_n']:
    m[c] = pd.to_numeric(m[c], errors='coerce')

m['ix_spend28_size'] = m['spend_28'] * m['size_n']
m['ix_spend28_kid'] = m['spend_28'] * m['kid_n']
m['ix_spend28_age'] = m['spend_28'] * m['age_n']
m['ix_ewma4_size'] = m['ewma_4'] * m['size_n']
m['ix_ewma4_kid'] = m['ewma_4'] * m['kid_n']
m['ix_tlag_size'] = m['tlag_mean'] * m['size_n']
m['ix_spend28_home'] = m['spend_28'] * m['home_n']
m['ix_size_ewma8'] = m['ewma_8'] * m['size_n']
print(m.shape)
agent_api.save_table(m, 'e013_full.parquet')
