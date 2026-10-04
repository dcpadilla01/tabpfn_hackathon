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
