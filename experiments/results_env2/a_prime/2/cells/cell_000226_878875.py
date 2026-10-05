
# Test: are outer-scope globals visible inside fn?
SENTINEL = 'visible'
def fn(view, sd):
    try:
        val = SENTINEL
    except NameError:
        val = 'NameError'
    return pd.DataFrame({'g_test': [val]}, index=view.households)

out = build_features(fn)
print('global test:', out.g_test.unique())

# Data details needed for reconstruction
s = snapshot(459)
t = s.transactions
print('\ntrans shape', t.shape)
print(t[['sales_value','coupon_disc','coupon_match_disc','retail_disc','quantity']].describe().round(3))
print('\ncampaigns head'); print(s.campaigns.head(8).to_string())
print('\ncampaign_targets head'); print(s.campaign_targets.head(5).to_string())
print('campaign_targets desc counts:'); print(s.campaign_targets.description.value_counts())
print('\ndemographics classification_1 uniques:', sorted(s.demographics.classification_1.dropna().unique())[:10])
print('homeowner:', s.demographics.homeowner_desc.value_counts().to_dict())
print('kid_cat:', s.demographics.kid_category_desc.value_counts().to_dict())
print('c4:', s.demographics.classification_4.value_counts().to_dict())
print('snapshot_days:', snapshot_days())
