import agent_api as A
import pandas as pd, numpy as np

print("snapshot days:", A.snapshot_days())
tt = A.train_targets()
print("train targets:", tt.shape)
print(tt.future_spend_4w.describe())
print("zero share:", (tt.future_spend_4w == 0).mean())

v = A.snapshot()
print("tx", v.transactions.shape)
print("demo", v.demographics.shape)
print("camp", v.campaigns.shape)
print("ctgt", v.campaign_targets.shape)
print("cred", v.coupon_redemptions.shape)
print("coup", v.coupons.shape)
print("dm", v.display_mailer.shape)
print("prod", v.products.shape)

tx = v.transactions
print("tx day range:", tx.day.min(), tx.day.max())
print("nunique hh in tx:", tx.household_key.nunique())
print("KEYS:", A.KEYS, "TARGET:", A.TARGET)
print(tx.head(3))


# ---- cell ----
import agent_api as A
import pandas as pd, numpy as np

def make(view, snap):
    tx = view.transactions
    hh = list(view.households)
    d = {'household_key': hh}
    df = pd.DataFrame(d).set_index('household_key')

    def agg(lo, hi):
        t = tx[(tx.day >= lo) & (tx.day <= hi)]
        g = t.groupby('household_key')
        return g.sales_value.sum(), g.basket_id.nunique(), g.day.nunique()

    def agg_all():
        t = tx
        g = t.groupby('household_key')
        return g.sales_value.sum(), g.basket_id.nunique(), g.day.nunique()

    s28, b28, dy28 = agg(snap-27, snap)
    s56, b56, _ = agg(snap-55, snap)
    s84, b84, _ = agg(snap-83, snap)
    s364, b364, _ = agg(max(1,snap-363), snap)
    sall, ball, dall = agg_all()

    df['spend_28'] = s28.reindex(df.index).fillna(0.0)
    df['spend_56'] = s56.reindex(df.index).fillna(0.0)
    df['spend_84'] = s84.reindex(df.index).fillna(0.0)
    df['spend_364'] = s364.reindex(df.index).fillna(0.0)
    df['spend_all'] = sall.reindex(df.index).fillna(0.0)
    df['baskets_28'] = b28.reindex(df.index).fillna(0.0)
    df['baskets_84'] = b84.reindex(df.index).fillna(0.0)
    df['days_28'] = dy28.reindex(df.index).fillna(0.0)
    df['basket_val_28'] = df['spend_28'] / df['baskets_28'].replace(0, np.nan)
    df['basket_val_84'] = df['spend_84'] / df['baskets_84'].replace(0, np.nan)
    df['recency'] = snap - tx.groupby('household_key').day.max().reindex(df.index)
    df['tenure'] = snap - tx.groupby('household_key').day.min().reindex(df.index)
    # prev-28 window spend for trend
    s_prev = agg(snap-55, snap-28)[0].reindex(df.index).fillna(0.0)
    df['trend_28'] = df['spend_28'] / s_prev.replace(0, np.nan)
    df['trend_28'] = df['trend_28'].replace([np.inf,-np.inf], np.nan)
    df['active_28'] = (df['spend_28'] > 0).astype(int)
    df['spend_28_div_84'] = df['spend_28'] / df['spend_84'].replace(0, np.nan)
    df['spend_28_div_364'] = df['spend_28'] / df['spend_364'].replace(0, np.nan)
    df['spend_364_per_28w'] = df['spend_364'] / 13.0
    df = df.replace([np.inf,-np.inf], np.nan)
    return df

out = A.build_features(make)
print(out.shape, out.columns.tolist())
print(out.head(3))
p = A.save_table(out, 'hist_v1')
print(p)
