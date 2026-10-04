import numpy as np, pandas as pd
base = load_saved('e005_full_plus_mix.parquet')
mix_cols = ['p_GROCERY','p_DRUG GM','p_PRODUCE','p_COSMETICS','p_NUTRITION','p_MEAT','p_MEAT-PCKGD','p_DELI','p_PASTRY','p_FLORAL','p_SEAFOOD-PCKGD','p_MISC. TRANS.','p_SPIRITS','p_SEAFOOD','p_other','p_private','unit_price','n_prod84','dow_entropy','modal_dow','zero_w12','wk_cv']
e3 = base.drop(columns=[c for c in mix_cols if c in base.columns]).copy()
print("E003-equiv feats:", e3.shape[1]-2)

out = pd.DataFrame({'household_key': e3.household_key, 'snapshot_day': e3.snapshot_day})
# log1p transforms of main spend levels
for c in ['spend_28','spend_56','spend_84','spend_364','spend_all','x_exp4w']:
    out['log1p_'+c] = np.log1p(e3[c].clip(lower=0))
# per-snapshot winsorized levels (99th pct of cross-section)
g = e3.groupby('snapshot_day')
for c in ['spend_84','spend_28','x_exp4w']:
    cap = g[c].transform(lambda s: s.quantile(0.99))
    out['win_'+c] = np.minimum(e3[c], cap)
# absolute department spends (share x spend_84)
for tag, col in [('gas','x_dep_KIOSK-GAS'),('grocery','x_dep_GROCERY'),('produce','x_dep_PRODUCE'),('meat','x_dep_MEAT'),('drug','x_dep_DRUG GM')]:
    out[tag+'_abs84'] = e3[col].fillna(0) * e3['spend_84'].fillna(0)
# log1p of absolutes
for c in ['gas_abs84','grocery_abs84']:
    out['log1p_'+c] = np.log1p(out[c].clip(lower=0))

e7 = pd.concat([e3.reset_index(drop=True), out.drop(columns=['household_key','snapshot_day']).reset_index(drop=True)], axis=1)
print("E007 table:", e7.shape)
print("new cols:", [c for c in out.columns if c not in ('household_key','snapshot_day')])
print("NaN check:", e7.isna().mean().max())
p = save_table(e7, 'e007_robust')
print("PATH:", p)