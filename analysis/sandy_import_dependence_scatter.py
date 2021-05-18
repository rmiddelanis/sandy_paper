import copy
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import tqdm
import pickle

from scipy import stats

from analysis.dataformat import AggrData
import matplotlib.pyplot as plt
import sys
from analysis.utils import WORLD_REGIONS

year = 2012

aggregation_time = 100

experiment_series_dir = '/mnt/cluster_temp/acclimate_run/scaling_ensembles/sandy_paper/SANDY_econYear2012_noSec_15000s_expRecovery_i1.0_1.0__d_1.0_1.0_elasticities/'

acclimate_datacap = pickle.load(open(experiment_series_dir + "/pickles/SANDY_2012_all_sec_t450__data_cap.pk", 'rb'))
acclimate_data = AggrData(acclimate_datacap)
consumption_data = acclimate_data.get_vars('consumption').get_sectors('FCON')
consumption_data.data_capsule.data = consumption_data.get_data() * 1000 #convert to single dollars
consumption_data = consumption_data.get_regions(list(set(consumption_data.get_regions()) - {'SDS', 'SDN'}))

consumption_difference = copy.deepcopy(consumption_data.clip(1))
consumption_difference.data_capsule.data = consumption_data.clip(aggregation_time).get_data().sum(axis=-1, keepdims=True) - consumption_data.clip(1).get_data() * aggregation_time

consumption_deviation = copy.deepcopy(consumption_data.clip(1))
consumption_deviation.data_capsule.data = consumption_data.clip(aggregation_time).get_data().sum(axis=-1, keepdims=True) / (consumption_data.clip(1).get_data() * aggregation_time) - 1

consumption_max = copy.deepcopy(consumption_data.clip(1))
consumption_max.data_capsule.data = consumption_data.clip(55).get_data().max(axis=-1, keepdims=True)

# trade_flows = pd.DataFrame(columns=['region', 'total_export', 'total_import', 'export_to_NJ', 'export_to_NY',
#                                     'import_from_NJ', 'import_from_NY'])
# trade_flows.set_index('region', inplace=True, drop=True)

# baseline = Dataset("/mnt/cluster_p/projects/acclimate/data/eora/EORA2012_CHN_USA.nc", 'r', format='NETCDF4')
# r_slice_nj = np.where(baseline['index_region'] == np.where(baseline['region'][:] == 'US.NJ')[0][0])[0]
# r_slice_ny = np.where(baseline['index_region'] == np.where(baseline['region'][:] == 'US.NY')[0][0])[0]
# for r in tqdm.tqdm(baseline['region'][:]):
#     if r in ['US.NJ', 'US.NY']:
#         continue
#     r_idx = np.where(baseline['region'][:] == r)[0][0]
#     r_slice = np.where(baseline['index_region'] == r_idx)[0]
#     r_outflow = baseline['flows'][r_slice, :].sum() - baseline['flows'][r_slice, r_slice].sum()
#     r_inflow = baseline['flows'][:, r_slice].sum() - baseline['flows'][r_slice, r_slice].sum()
#     r_outflow_to_nj = baseline['flows'][r_slice, r_slice_nj].sum()
#     r_outflow_to_ny = baseline['flows'][r_slice, r_slice_ny].sum()
#     r_inflow_from_nj = baseline['flows'][r_slice_nj, r_slice].sum()
#     r_inflow_from_ny = baseline['flows'][r_slice_ny, r_slice].sum()
#     trade_flows.loc[r] = [r_outflow, r_inflow, r_outflow_to_nj, r_outflow_to_ny, r_inflow_from_nj, r_inflow_from_ny]

# trade_flows['nj_import_share'] = trade_flows['import_from_NJ'] / trade_flows['total_import']
# trade_flows['ny_import_share'] = trade_flows['import_from_NY'] / trade_flows['total_import']
# trade_flows['nj_export_share'] = trade_flows['export_to_NJ'] / trade_flows['total_export']
# trade_flows['ny_export_share'] = trade_flows['export_to_NY'] / trade_flows['total_export']
# trade_flows['ny_trade_balance'] = trade_flows['export_to_NY'] - trade_flows['import_from_NY']
# trade_flows['nj_trade_balance'] = trade_flows['export_to_NJ'] - trade_flows['import_from_NJ']

trade_flows = pd.DataFrame(columns=['region', 'total_export', 'total_import', 'export_to_US', 'import_from_US'])
trade_flows.set_index('region', inplace=True, drop=True)

baseline = Dataset("/mnt/cluster_p/projects/acclimate/data/eora/EORA2012.nc", 'r', format='NETCDF4')
r_idx_usa = np.where(baseline['region'][:] == 'USA')[0][0]
for r in tqdm.tqdm(baseline['region'][:]):
    if r in ['USA']:
        continue
    r_idx = np.where(baseline['region'][:] == r)[0][0]
    r_outflow = baseline['flows'][:, r_idx, ...].sum() - baseline['flows'][:, r_idx, :, r_idx].sum()
    r_inflow = baseline['flows'][..., :, r_idx].sum() - baseline['flows'][:, r_idx, :, r_idx].sum()
    r_outflow_to_us = baseline['flows'][:, r_idx, :, r_idx_usa].sum()
    r_inflow_from_us = baseline['flows'][:, r_idx_usa, :, r_idx].sum()
    trade_flows.loc[r] = [r_outflow, r_inflow, r_outflow_to_us, r_inflow_from_us]

trade_flows['us_import_share'] = trade_flows['import_from_US'] / trade_flows['total_import']
trade_flows['us_export_share'] = trade_flows['export_to_US'] / trade_flows['total_export']
trade_flows['us_trade_balance'] = trade_flows['export_to_US'] - trade_flows['import_from_US']

consumption_df = pd.DataFrame(columns=['region', 'consumption_difference', 'absolute_consumption_difference',
                                       'relative_consumption_loss', 'consumption_maximum'])
consumption_df.set_index('region', drop=True)
for r in consumption_data.get_regions():
    consumption_df.loc[r] = [r, consumption_difference.get_regions(r).get_data().flatten()[0],
                             abs(consumption_difference.get_regions(r).get_data().flatten()[0]),
                             consumption_deviation.get_regions(r).get_data().flatten()[0],
                             consumption_max.get_regions(r).get_data().flatten()[0]]

plot_data = trade_flows.join(consumption_df, how='inner')
# plot_data = plot_data.loc[(~plot_data.index.isin(WORLD_REGIONS['USA'])) & (~plot_data.index.isin(WORLD_REGIONS['CHN']))]
# plot_data = plot_data.loc[plot_data.index.isin(WORLD_REGIONS['USA'])]
# plot_data = plot_data.loc[plot_data.index.isin(WORLD_REGIONS['CHN'])]

##### LOG-LOG scale #####
# x_var = 'export_to_NJ'
x_var = 'import_from_US'
y_var = 'absolute_consumption_difference'
# z_var = 'nj_trade_balance'
z_var = 'us_trade_balance'
# min_flow = 1e9 / 365 # EORA data is per days (probably?)
min_flow = 0
s_min, s_max = 50, 1000
s_offset = min(plot_data[z_var])
s_scale = s_max / (max(plot_data[z_var]) - min(plot_data[z_var]))
gains_x = []
gains_y = []
losses_x = []
losses_y = []
for r in plot_data.index:
    if plot_data.loc[r, x_var] >= min_flow:
        scatter_size = s_min + (plot_data.loc[r, z_var] - s_offset) * s_scale
        color = 'g' if plot_data.loc[r, z_var] > 0 else 'r'
        x, y = plot_data.loc[r, x_var], plot_data.loc[r, y_var]
        log_x = np.log(x) if x > 0 else -np.log(-x)
        log_y = np.log(y) if y > 0 else -np.log(-y)
        if log_y > 0:
            gains_x += [log_x]
            gains_y += [log_y]
        else:
            losses_x += [log_x]
            losses_y += [log_y]
        plt.scatter(log_x, log_y, s=scatter_size, color=color, alpha=0.3)
        plt.annotate(r, (log_x, log_y))
reg_gains = stats.linregress(gains_x, gains_y)
plt.plot([min(gains_x), max(gains_x)], [reg_gains.intercept + reg_gains.slope * min(gains_x), reg_gains.intercept + reg_gains.slope * max(gains_y)], 'r')
reg_losses = stats.linregress(losses_x, losses_y)
plt.plot([min(losses_x), max(losses_x)], [reg_losses.intercept + reg_losses.slope * min(losses_x), reg_losses.intercept + reg_losses.slope * max(losses_x)], 'r')

##### log log scale#####
fig, ax = plt.subplots()
ax.set_yscale('symlog')
ax.set_xscale('symlog')
##### linear scale #####
# x_var = 'export_to_NJ'
x_var = 'export_to_US'
y_var = 'consumption_difference'
# z_var = 'nj_trade_balance'
z_var = 'relative_consumption_loss'
# min_flow = 1e9 / 365 # EORA data is per days (probably?)
min_flow = -1e100
s_min, s_max = 50, 1500
s_offset = min(abs(plot_data[z_var]))
s_scale = s_max / (max(abs(plot_data[z_var])) - min(abs(plot_data[z_var])))
for r in plot_data.index:
    if plot_data.loc[r, x_var] >= min_flow:
        scatter_size = s_min + (abs(plot_data.loc[r, z_var]) - s_offset) * s_scale
        color = 'g' if plot_data.loc[r, z_var] > 0 else 'r'
        plt.scatter(plot_data.loc[r, x_var], plot_data.loc[r, y_var], s=scatter_size, color=color, alpha=0.3)
        plt.annotate(r, (plot_data.loc[r, x_var], plot_data.loc[r, y_var]), fontsize=5)
ax.set_xlabel('US trade balance (USD)')
ax.set_ylabel('consumption difference (USD)')

##### linear scale #####
# x_var = 'export_to_NJ'
x_var = 'import_from_US'
y_var = 'absolute_consumption_difference'
# z_var = 'nj_trade_balance'
z_var = 'us_trade_balance'
# min_flow = 1e9 / 365 # EORA data is per days (probably?)
min_flow = 0
s_min, s_max = 50, 1000
s_offset = min(plot_data[z_var])
s_scale = s_max / (max(plot_data[z_var]) - min(plot_data[z_var]))
for r in plot_data.index:
    if plot_data.loc[r, x_var] >= min_flow:
        scatter_size = s_min + (plot_data.loc[r, z_var] - s_offset) * s_scale
        color = 'g' if plot_data.loc[r, z_var] > 0 else 'r'
        plt.scatter(plot_data.loc[r, x_var], plot_data.loc[r, y_var], s=scatter_size, color=color, alpha=0.3)
        plt.annotate(r, (plot_data.loc[r, x_var], plot_data.loc[r, y_var]))