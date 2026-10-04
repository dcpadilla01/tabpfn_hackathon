
import agent_api, pandas as pd, numpy as np

base = agent_api.load_saved('e012_basket_shape.parquet')
cand = agent_api.load_saved('cand_screen1.parquet')
b = base.set_index(['household_key','snapshot_day'])

# curated blocks
keep_cand = ['c_dev28','c_dev84','c_cv26','c_burst7','c_gap_trend']
A = pd.DataFrame(index=b.index)
rec = b['recency']; gm = b['gap_mean_84']; zf = b['zero_frac_13']; s28 = b['spend_28']; w8a = b['wk_avg_8']
A['z_rec45'] = (rec > 45).astype(float)
A['z_gap30'] = (gm > 30).astype(float)
A['z_zf50']  = (zf > 0.5).astype(float)
wks = b[['w3','w4','w5','w6','w7','w8']]
A['t_maxwk'] = wks.max(axis=1)
A['t_p90wk'] = wks.quantile(0.9, axis=1)
A['t_top2']  = (b['w7']+b['w8'])/np.maximum(wks.sum(axis=1),1e-6)
C = pd.DataFrame(index=b.index)
ten = b['tenure'].clip(lower=0)
C['c_gate28'] = s28*ten/(ten+50)
C['c_gate84'] = b['spend_84']*ten/(ten+100)
C['c_shr28'] = (s28*ten + 130*50)/(ten+50)
C['c_shr84'] = (b['spend_84']*ten + 130*200)/(ten+200)
C['c_lograte'] = np.log1p(b['spend_rate_life'])
N = pd.DataFrame(index=b.index)
s84 = b['spend_84']; tr = b['trips_28']; fwm = b['fwd28_mean']
N['n_rank84'] = s84.groupby(level=1).rank(pct=True)
N['n_rank28'] = s28.groupby(level=1).rank(pct=True)
N['n_rankfwm'] = fwm.groupby(level=1).rank(pct=True)
N['n_ranktr'] = tr.groupby(level=1).rank(pct=True)
N['n_sqrt28'] = np.sqrt(np.clip(s28,0,None))
N['n_log_sq'] = np.log1p(s28)**2

blocks = [base,
          cand[keep_cand].reset_index(),
          A.reset_index(), C.reset_index(), N.reset_index()]
out = blocks[0]
for blk in blocks[1:]:
    out = out.merge(blk, on=['household_key','snapshot_day'], how='inner')
print(out.shape)
assert out.shape[0] == 36426
newcols = [c for c in out.columns if c not in base.columns]
print(len(newcols), newcols)
agent_api.save_table(out, 'e013_denoise')
