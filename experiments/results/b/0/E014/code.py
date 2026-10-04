import pandas as pd, numpy as np
v = agent_api.snapshot()
t = v.transactions
print("TXN", t.shape)
print("neg:", int((t.sales_value<0).sum()), "zero:", int((t.sales_value==0).sum()))
print("day range", int(t.day.min()), int(t.day.max()))
cr = v.coupon_redemptions
print("CR", cr.shape); print(cr.head(3))
dm = v.display_mailer
print("DM", dm.shape); print(dm.head(3))
print("display:", dm.display.value_counts().sort_index().to_dict())
print("mailer:", dm.mailer.value_counts().sort_index().to_dict())
h = v.households
print("households", type(h), h.shape)
print(h.head(3))
print("week", v.week, "day", v.day)
e11 = agent_api.load_saved('e011_price.parquet')
print("E011", e11.shape)
print(list(e11.columns))

# ---- cell ----
import pandas as pd, numpy as np
v = agent_api.snapshot()
print("week", v.week, "day", v.day)
print("KEYS", agent_api.KEYS, "TARGET", agent_api.TARGET)
e11 = agent_api.load_saved('e011_price.parquet')
print("E011", e11.shape)
print(list(e11.columns))
print(e11.head(2).T.head(30))

# ---- cell ----
import pandas as pd, numpy as np

tt = agent_api.train_targets()
y = tt.future_spend_4w
print("train rows", len(tt), "mean", y.mean(), "median", y.median(), "zero share", (y==0).mean())
print(y.describe())

def probe(view, sd):
    hh = view.households
    print("sd", sd, "hh type", type(hh), "len", len(hh))
    if sd == 95:
        print("hh head:", list(hh)[:5])
        t = view.transactions
        print("txn max day", int(t.day.max()), "dm max week", int(view.display_mailer.week_no.max()))
        print("week attr", view.week, "computed wk", (sd+8)//7)
        print("hh indexable?", hh[0] if hasattr(hh,'__getitem__') else 'no getitem')
    return pd.DataFrame({"x": np.arange(len(hh), dtype=float)}, index=hh)

bf = agent_api.build_features(probe)
print("build_features rows", len(bf), "cols", list(bf.columns))
print(bf.groupby('snapshot_day').size())

# ---- cell ----
import pandas as pd, numpy as np
v = agent_api.snapshot()
t = v.transactions; dm = v.display_mailer
print(t.dtypes.to_dict()); print(dm.dtypes.to_dict())
chk = ((t.week_no == (t.day+8)//7).mean())
print("week_no consistent:", chk)
tg = t.groupby(['product_id','store_id','week_no']).size().reset_index(name='n')
print("tx combos", len(tg), "stores", t.store_id.nunique(), "prods", t.product_id.nunique())
mg = tg.merge(dm, on=['product_id','store_id','week_no'], how='left')
print("match rate", round(mg.display.notna().mean(),4))
sub = mg[mg.display.notna()]
dmap = {c:i for i,c in enumerate(['0','1','2','3','4','5','6','7','9','A'])}
mmap = {c:i for i,c in enumerate(['0','A','C','D','F','H','J','L','P','X','Z'])}
sub2 = sub.copy()
sub2['dv'] = sub2.display.map(dmap); sub2['mv'] = sub2.mailer.map(mmap)
print("dv dist", sub2.dv.value_counts().sort_index().to_dict())
print("mv dist", sub2.mv.value_counts().sort_index().to_dict())
print("display>0 share among matched:", round((sub2.dv>0).mean(),4), "mailer>0:", round((sub2.mv>0).mean(),4))

# ---- cell ----
import pandas as pd, numpy as np
v = agent_api.snapshot(459)
t = v.transactions; dm = v.display_mailer
print("dm dup combos:", int(dm.duplicated(['product_id','store_id','week_no']).sum()), "dm rows", len(dm))
dmap = {c:i for i,c in enumerate(['0','1','2','3','4','5','6','7','9','A'])}
mmap = {c:i for i,c in enumerate(['0','A','C','D','F','H','J','L','P','X','Z'])}
# aggregate txn to product-store-week
tg = t.groupby(['product_id','store_id','week_no'], as_index=False).agg(
    spend=('sales_value','sum'), qty=('quantity','sum'), nlines=('sales_value','size'),
    hh=('household_key','nunique'))
mg = tg.merge(dm, on=['product_id','store_id','week_no'], how='left')
mg['dv'] = mg.display.map(dmap).astype('float64')
mg['mv'] = mg.mailer.map(mmap).astype('float64')
mg['dv'] = mg.dv.fillna(0.0); mg['mv'] = mg.mv.fillna(0.0)
mg['disp'] = (mg.dv>0).astype(float); mg['mail'] = (mg.mv>0).astype(float)
print("matched share of combos:", round(mg.display.notna().mean(),4))
print("exposed spend share:", round(mg.loc[mg.disp>0,'spend'].sum()/mg.spend.sum(),4),
      "mailer:", round(mg.loc[mg.mail>0,'spend'].sum()/mg.spend.sum(),4))
print("dv>0 among matched:", round((mg.loc[mg.display.notna(),'dv']>0).mean(),4),
      "mv>0:", round((mg.loc[mg.display.notna(),'mv']>0).mean(),4))
# per-household join timing
hh = v.households
line = t[['household_key','product_id','store_id','week_no']].merge(
    mg[['product_id','store_id','week_no','spend','qty','nlines','dv','mv','disp','mail']],
    on=['product_id','store_id','week_no'], how='left').fillna(0.0)
g = line.groupby('household_key').agg(dm_spend=('spend','sum'), dm_qty=('qty','sum'),
      dm_trips=('nlines','sum'), dm_nprod=('product_id','nunique'),
      dm_disp_spend=('spend', lambda s: 0.0), dm_mail_spend=('spend','sum'))
# proper disp/mail sums via precomputed columns
line['dsp'] = line.spend*line.disp; line['msp'] = line.spend*line.mail
g2 = line.groupby('household_key').agg(dm_spend=('spend','sum'), dm_qty=('qty','sum'),
      dm_trips=('nlines','sum'), dm_nprod=('product_id','nunique'),
      dm_disp_spend=('dsp','sum'), dm_mail_spend=('msp','sum'))
print(g2.describe())
print("hh coverage:", g2.index.isin(hh).mean() if hasattr(g2.index,'isin') else 'n/a')
print("spend28 check: total spend all households", t.sales_value.sum(), "dm_spend sum", g2.dm_spend.sum())

# ---- cell ----
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