import pandas as pd
import os
import us
import numpy as np

asset_naics_rename = {
    'Private fixed assets': 'All industry total',
    'Agriculture, forestry, fishing, and hunting': 'Agriculture, forestry, fishing and hunting',
    'Other services, except government': 'Other services (except government and government enterprises)',
    'Administrative and waste management services': 'Administrative and support and waste management and remediation services',
    'Health and social assistance': 'Health care and social assistance',
    'Mining': 'Mining, quarrying, and oil and gas extraction',
    'Government fixed assets': 'Government and government enterprises',
}

naics_sectors = [
    'All industry total',
    'Agriculture, forestry, fishing and hunting',
    'Mining, quarrying, and oil and gas extraction',
    'Utilities',
    'Construction',
    'Manufacturing',
    'Wholesale trade',
    'Retail trade',
    'Transportation and warehousing',
    'Information',
    'Finance and insurance',
    'Real estate and rental and leasing',
    'Professional, scientific, and technical services',
    'Management of companies and enterprises',
    'Administrative and support and waste management and remediation services',
    'Educational services',
    'Health care and social assistance',
    'Arts, entertainment, and recreation',
    'Accommodation and food services',
    'Other services (except government and government enterprises)',
    'Government and government enterprises',
]

us_state_names = [state.name for state in us.states.STATES]
us_state_abbrvs = [us.states.mapping('name', 'abbr')[name] for name in us_state_names]

state_name_mapping = {'United States *': 'USA'}
for us_state_name in us_state_names:
    state_name_mapping[us_state_name] = us.states.mapping('name', 'abbr')[us_state_name]

# Data from: https://apps.bea.gov/regional/downloadzip.cfm
# SAGDP2N: GDP in Current Dollars by Counties and MSA
# values are in millions of USD
state_gdp = pd.read_csv("external/SAGDP2N__ALL_AREAS_1997_2020.csv", engine='python', na_values=['(NA)', '(D)', '(L)'])
state_gdp.drop('2020', axis=1, inplace=True)
state_gdp.drop(state_gdp.index[-4:], inplace=True)
state_gdp.dropna(inplace=True)
state_gdp = state_gdp.drop(['GeoFIPS', 'Region', 'TableName', 'LineCode', 'IndustryClassification', 'Unit'], axis=1)
state_gdp.iloc[:, 2:] = state_gdp.iloc[:, 2:].astype(float)
state_gdp['Description'] = state_gdp['Description'].apply(lambda x: x.lstrip(' '))
state_gdp['GeoName'] = state_gdp['GeoName'].apply(lambda x: x[:-1].rstrip(' ') if x[-1] == '*' else x)
state_gdp = state_gdp.drop(state_gdp.index[~state_gdp['GeoName'].isin(us_state_names + ['United States'])]).reset_index(drop=True)
state_gdp['GeoName'] = state_gdp['GeoName'].apply(lambda x: 'USA' if x == 'United States' else us.states.lookup(x).abbr)
state_gdp = state_gdp.rename({'GeoName': 'State', 'Description': 'Sector', 'GeoName': 'State'}, axis=1)
assert len(set(state_gdp['State'].values)) == len(state_name_mapping)
state_gdp = state_gdp[state_gdp['Sector'].isin(naics_sectors)]
state_gdp.set_index(['State', 'Sector'], inplace=True, drop=True)
state_gdp.columns = [int(col) for col in state_gdp.columns]


# Data from: https://apps.bea.gov/iTable/iTable.cfm?ReqID=10&step=2
# Current-Cost Net Stock of Private Fixed Assets by Industry
# Current-Cost Net Stock of Fixed Assets and Consumer Durable Goods
# values are in billions uf USD --> transform to millions of USD
asset_tables_register = pd.read_csv('../data/external/assets/TablesRegister.txt')
asset_series_register = pd.read_csv('../data/external/assets/SeriesRegister.txt')
asset_data = pd.read_csv('../data/external/assets/FixedAssets.txt', dtype={'Value': np.float64}, thousands=',')
private_assets_title = 'Table 3.1ESI. Current-Cost Net Stock of Private Fixed Assets by Industry'
nonprivate_assets_title = 'Table 1.1. Current-Cost Net Stock of Fixed Assets and Consumer Durable Goods'
private_assets_id = asset_tables_register[asset_tables_register['TableTitle'] == private_assets_title]['TableId'].iloc[0]
nonprivate_assets_id = asset_tables_register[asset_tables_register['TableTitle'] == nonprivate_assets_title]['TableId'].iloc[0]
private_assets_df = asset_series_register[asset_series_register['TableId:LineNo'].str.contains(private_assets_id)][['SeriesLabel', '%SeriesCode']]
nonprivate_assets_df = asset_series_register[asset_series_register['TableId:LineNo'].str.contains(nonprivate_assets_id)][['SeriesLabel', '%SeriesCode']]
assets_df = pd.concat((private_assets_df, nonprivate_assets_df))
assets_df = assets_df.drop_duplicates().set_index('%SeriesCode', drop=True)
assets_df = assets_df.join(asset_data.pivot(index='%SeriesCode', columns='Period', values='Value'))
assets_df = assets_df.rename({'SeriesLabel': 'Sector'}, axis=1)
assets_df['Sector'] = assets_df['Sector'].apply(lambda x: asset_naics_rename[x] if x in asset_naics_rename.keys() else x)
assets_df = assets_df[assets_df['Sector'].isin(naics_sectors)]
assets_df = assets_df.set_index('Sector', drop=True)
assets_df.iloc[:, 1:] = assets_df.iloc[:, 1:] * 1e3

common_years = list(set(assets_df.columns) & set(state_gdp.columns))
asset_gdp_ratio = assets_df[common_years].sort_index() / state_gdp[common_years].loc['USA'].sort_index()
asset_gdp_ratio = asset_gdp_ratio.rolling(window=10, axis=1, center=True, min_periods=1).mean()

asset_gdp_ratio.to_csv('generated/asset_gdp_ratio_by_sector.csv', sep=',', na_rep=[''])
state_gdp.to_csv('generated/gdp_by_state.csv', sep=',', na_rep=[''])
