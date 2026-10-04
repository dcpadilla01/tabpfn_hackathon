t = agent_api.load_saved('rich_behavioral.parquet')
print("shape:", t.shape)
print("columns:", list(t.columns))
print(t.dtypes.value_counts())
print("snapdays:", agent_api.snapshot_days())
s = agent_api.snapshot()
dm = s.display_mailer
print("\ndisplay_mailer shape:", dm.shape)
print(dm.head(3))
print("display vals:", dm['display'].value_counts().head().to_dict())
print("mailer vals:", dm['mailer'].value_counts().head().to_dict())
ct = s.campaign_targets
print("\ncampaign_targets shape:", ct.shape)
print(ct.head(3))
print(ct['description'].value_counts().to_dict())
cr = s.coupon_redemptions
print("\ncoupon_redemptions shape:", cr.shape)
print(cr.head(3))
camps = s.campaigns
print("\ncampaigns shape:", camps.shape)
print(camps.head(5))


# ---- cell ----
import numpy as np, pandas as pd
t = agent_api.load_saved('rich_behavioral.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
print("train rows:", len(df))
y = df['future_spend_4w']
print("target stats:", y.describe().to_dict())
print("target by snapshot day:")
print(df.groupby('snapshot_day')['future_spend_4w'].agg(['mean','median','count']))
feats = [c for c in t.columns if c not in ('household_key','snapshot_day')]
corr = df[feats+['future_spend_4w']].corr()['future_spend_4w'].drop('future_spend_4w').sort_values(key=abs, ascending=False)
print("\ntop |corr| with target:")
print(corr.head(20))
print("\nlowest:")
print(corr.tail(10))


# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('rich_behavioral.parquet')

def make(view, day):
    w = (day + 8) // 7
    tr = view.transactions
    popw = tr.groupby('week_no')['sales_value'].sum()

    def prof(v):
        # seasonal share of week v relative to trailing 52 weeks
        if v < 1 or v not in popw.index:
            return 1.0
        prev = [u for u in range(max(1, v - 52), v) if u in popw.index]
        if not prev:
            return 1.0
        base = popw.loc[prev].mean()
        return popw[v] / base if base > 0 else 1.0

    # lift for coming 4 weeks, read from same weeks one year earlier
    lifts = [prof(v - 52) if v - 52 >= 1 else 1.0 for v in range(w + 1, w + 5)]
    lift = float(np.mean(lifts))

    hh = view.households
    out = pd.DataFrame(index=hh)

    # household spend in the same 4-week window one year ago (day-aligned)
    d0, d1 = day - 363, day - 336
    if d0 >= 1:
        sub = tr[(tr.day >= d0) & (tr.day <= d1)]
        out['own_ly_spend4w'] = sub.groupby('household_key')['sales_value'].sum()
    else:
        out['own_ly_spend4w'] = 0.0
    # two years ago
    e0, e1 = day - 727, day - 700
    if e0 >= 1:
        sub2 = tr[(tr.day >= e0) & (tr.day <= e1)]
        out['own_ly2_spend4w'] = sub2.groupby('household_key')['sales_value'].sum()
    else:
        out['own_ly2_spend4w'] = 0.0

    out['season_lift'] = lift
    wk = (w + 2.0)
    out['week_sin'] = np.sin(2 * np.pi * wk / 52.0)
    out['week_cos'] = np.cos(2 * np.pi * wk / 52.0)
    return out

feats = agent_api.build_features(make)
print("built:", feats.shape, feats['season_lift'].describe().to_dict())
merged = base.merge(feats.reset_index().rename(columns={'index': 'household_key'}),
                    on=['household_key', 'snapshot_day'], how='inner')
print("merged:", merged.shape)
print("nulls:", merged.isnull().sum().sum())
path = agent_api.save_table(merged, 'season.parquet')
print("saved:", path)

# ---- cell ----
import numpy as np, pandas as pd

base = agent_api.load_saved('rich_behavioral.parquet')

def make(view, day):
    w = (day + 8) // 7
    tr = view.transactions
    popw = tr.groupby('week_no')['sales_value'].sum()

    def prof(v):
        if v < 1 or v not in popw.index:
            return 1.0
        prev = [u for u in range(max(1, v - 52), v) if u in popw.index]
        if not prev:
            return 1.0
        b = popw.loc[prev].mean()
        return popw[v] / b if b > 0 else 1.0

    lifts = [prof(v - 52) if v - 52 >= 1 else 1.0 for v in range(w + 1, w + 5)]
    lift = float(np.mean(lifts))

    hh = view.households
    out = pd.DataFrame(index=hh)
    d0, d1 = day - 363, day - 336
    if d0 >= 1:
        sub = tr[(tr.day >= d0) & (tr.day <= d1)]
        out['own_ly_spend4w'] = sub.groupby('household_key')['sales_value'].sum()
    else:
        out['own_ly_spend4w'] = 0.0
    e0, e1 = day - 727, day - 700
    if e0 >= 1:
        sub2 = tr[(tr.day >= e0) & (tr.day <= e1)]
        out['own_ly2_spend4w'] = sub2.groupby('household_key')['sales_value'].sum()
    else:
        out['own_ly2_spend4w'] = 0.0
    out['season_lift'] = lift
    wk = w + 2.0
    out['week_sin'] = np.sin(2 * np.pi * wk / 52.0)
    out['week_cos'] = np.cos(2 * np.pi * wk / 52.0)
    return out

feats = agent_api.build_features(make)
feats = feats.reset_index(drop=True)
print("built:", feats.shape)
merged = base.merge(feats, on=['household_key', 'snapshot_day'], how='inner')
print("merged:", merged.shape, "nulls:", int(merged.isnull().sum().sum()))
path = agent_api.save_table(merged, 'season.parquet')
print("saved:", path)