import pandas as pd, numpy as np

DMAP = {c:i for i,c in enumerate(['0','1','2','3','4','5','6','7','9','A'])}
MMAP = {c:i for i,c in enumerate(['0','A','C','D','F','H','J','L','P','X','Z'])}

def fn(view, sd):
    t = view.transactions
    dm = view.display_mailer
    wk = (sd + 8) // 7
    weeks_needed = list(range(max(wk-16,1), wk+1)) + list(range(max(wk-53,1), max(wk-47,1)))
    dms = dm[dm.week_no.isin(weeks_needed)].drop_duplicates(
        ['product_id','store_id','week_no'], keep='last')
    tg = t.groupby(['product_id','store_id','week_no'], as_index=False).agg(
        spend=('sales_value','sum'))
    mg = tg.merge(dms, on=['product_id','store_id','week_no'], how='left')
    mg['disp'] = mg.display.isin(['1','2','3','4','5','6','7','9','A']).astype('float64')
    mg['mail'] = mg.mailer.isin(['A','C','D','F','H','J','L','P','X','Z']).astype('float64')
    mg['disp'] = mg.disp.fillna(0.0); mg['mail'] = mg.mail.fillna(0.0)
    keep = ['product_id','store_id','week_no','disp','mail']
    lines = t[['household_key','day','product_id','store_id','week_no','sales_value']].merge(
        mg[keep], on=['product_id','store_id','week_no'], how='left')
    lines['disp'] = lines.disp.fillna(0.0); lines['mail'] = lines.mail.fillna(0.0)
    lines['dsp'] = lines.sales_value * lines.disp
    lines['msp'] = lines.sales_value * lines.mail
    idx = pd.Index(list(view.households), name='household_key')

    def wagg(lo, hi):
        m = (lines.day > lo) & (lines.day <= hi)
        g = lines.loc[m].groupby('household_key').agg(
            s=('sales_value','sum'), dsp=('dsp','sum'), msp=('msp','sum'))
        return g.reindex(idx)

    out = pd.DataFrame(index=idx)
    # 28d and 112d windows
    for name, w in [('28', 28), ('112', 112)]:
        g = wagg(sd - w, sd)
        s = g.s.fillna(0.0)
        out['dm_disp_spend' + name] = g.dsp.fillna(0.0)
        out['dm_mail_spend' + name] = g.msp.fillna(0.0)
        sdnz = s.replace(0.0, np.nan)
        out['dm_disp_share' + name] = g.dsp / sdnz
        out['dm_mail_share' + name] = g.msp / sdnz
    out['dm_disp_spend28_log'] = np.log1p(out['dm_disp_spend28'])
    out['dm_mail_spend28_log'] = np.log1p(out['dm_mail_spend28'])
    # year-ago same 4-week window (sd-364, sd-336]
    if sd - 364 >= 1:
        g = wagg(sd - 364, sd - 336)
        sdz = g.s.replace(0.0, np.nan)
        out['dm_disp_share_ly'] = g.dsp / sdz
        out['dm_mail_share_ly'] = g.msp / sdz
    else:
        out['dm_disp_share_ly'] = np.nan
        out['dm_mail_share_ly'] = np.nan
    # prior-period windows
    g = wagg(sd - 32, sd - 4)
    sdz = g.s.replace(0.0, np.nan)
    out['dm_disp_spend_p4'] = g.dsp.fillna(0.0)
    out['dm_mail_share_p4'] = g.msp / sdz
    g = wagg(sd - 3, sd)
    sdz = g.s.replace(0.0, np.nan)
    out['dm_disp_spend_f4'] = g.dsp.fillna(0.0)
    out['dm_mail_share_f4'] = g.msp / sdz
    return out

bf = agent_api.build_features(fn)
print("rows", len(bf), "cols", list(bf.columns))
print(bf.groupby('snapshot_day').size().to_dict())
num = bf.drop(columns=['household_key','snapshot_day'])
print("NaN share per feat:")
print((num.isna().mean()).round(3).to_dict())
print(num.describe().T[['mean','50%','75%']].round(3).to_string())
agent_api.save_table(bf, 'dm_exp.parquet')
print("saved dm_exp")