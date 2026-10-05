
import pandas as pd, numpy as np, agent_api, warnings
warnings.filterwarnings('ignore')

t = agent_api.train_targets()
v = agent_api.snapshot(as_of_day=431)
tx = v.table('transactions')
y431 = t[t.snapshot_day==431].set_index('household_key')['future_spend_4w']
hh = y431.index

def win_spend(tx, hh, lo, hi):
    d = tx[(tx.day>lo)&(tx.day<=hi)&(tx.household_key.isin(hh))]
    return d.groupby('household_key')['sales_value'].sum().reindex(hh).fillna(0.0)

s28 = win_spend(tx, hh, 403, 431)
y = y431.values

# --- Hypothesis A: per-household multiplicative drift, pooled shrinkage ---
df = pd.DataFrame({'y':y,'s28':s28.values})
df['hh'] = hh.values
# historical realized ratio: future4w / prior28d, using earlier train snapshots? We only have y at train snapshots.
# Instead estimate from data directly: for each household, ratio of (431->459 spend) is unknown at 431.
# Use "next 4w / last 4w" computed on PAST days (pseudo-targets): for day d in 336..403, y_pseudo(d)=spend(d+1..d+28), x=spend(d-27..d)
# do it for a few d to estimate per-household ratio with shrinkage
pseudo = []
for d in [347, 375, 403]:
    yp = win_spend(tx, hh, d+1, d+28)
    xp = win_spend(tx, hh, d-27, d)
    pseudo.append(pd.DataFrame({'hh':hh.values,'d':d,'yp':yp.values,'xp':xp.values}))
ps = pd.concat(pseudo)
ps['ratio'] = ps.yp/ps.xp.clip(lower=1e-6)
# per-household median ratio over pseudo windows (only where xp>0)
g = ps[ps.xp>10].groupby('hh')['ratio'].median()
print("pseudo-ratio distribution:", g.describe().round(3).to_dict())
print("share ratio>1.5:", (g>1.5).mean().round(3), " share ratio<0.67:", (g<0.667).mean().round(3))

# how predictive is per-household ratio (shrunken) for y at 431?
df = df.merge(g.rename('hratio'), left_on='hh', right_index=True, how='left')
df['hratio'] = df['hratio'].fillna(1.0)
# shrink toward 1
for lam in [0, 1, 3, 10]:
    r = (df.hratio*g.size + lam*1.0)/(g.size+lam)
    pred = df.s28 * ((df.hratio*g.size + lam*1.0)/(g.size+lam))
    print(f"lam={lam}: MAE={np.mean(np.abs(pred-y)):.2f}  corr={np.corrcoef(pred,y)[0,1]:.3f}")
print("baseline MAE s28:", np.mean(np.abs(df.s28-y)))
