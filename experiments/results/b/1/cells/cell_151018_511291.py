import agent_api, pandas as pd, numpy as np

t = agent_api.load_saved('e001_history.parquet')
tt = agent_api.train_targets()
df = t.merge(tt, on=['household_key','snapshot_day'], how='inner')
y = df.future_spend_4w

def mae(p): return np.abs(p - y).mean()
print('mean pred', mae(np.full(len(y), y.mean())))
print('median pred', mae(np.full(len(y), y.median())))
for c in ['spend_7','spend_28','spend_56','spend_84','spend_182','spend_365','spend_all','weekly_rate_84','spend_prev28']:
    print(f'persist {c:15s} MAE={mae(df[c].fillna(0).values):8.3f}')

# blend of spends via ridge
from numpy.linalg import lstsq
X = df[['spend_7','spend_28','spend_56','spend_84','spend_182','spend_365']].fillna(0).values
w = lstsq(X, y.values, rcond=None)[0]
print('ridge-ish coefs', np.round(w,3), 'MAE', mae(X@w))

# ratio target/spend_84
r = y / df.spend_84.replace(0, np.nan)
print('ratio quantiles', r.quantile([.1,.25,.5,.75,.9]).round(3).to_dict())

# binned: target vs spend_84
b = pd.qcut(df.spend_84, 10, duplicates='drop')
g = df.groupby(b, observed=True).agg(y=('future_spend_4w','mean'), s84=('spend_84','mean'), n=('future_spend_4w','size'))
print(g.round(1))