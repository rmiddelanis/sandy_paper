import copy
import os

import tqdm
from matplotlib import transforms
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from scipy.interpolate import interp1d
from netCDF4 import Dataset
from scipy import stats
from sklearn.linear_model import LinearRegression

from yellowbrick.regressor import ResidualsPlot
import matplotlib as mpl

mpl.rcdefaults()

from dataformat import AggrData
from utils import WORLD_REGIONS

from map import make_map, create_colormap

MAX_COLUMN_HEIGHT = 9.60
MAX_FIG_WIDTH_WIDE = 7.07
MAX_FIG_WIDTH_NARROW = 3.45
FSIZE_TINY = 6
FSIZE_SMALL = 8
FSIZE_MEDIUM = 10
FSIZE_LARGE = 12

plt.rc('font', size=FSIZE_SMALL)  # controls default text sizes
plt.rc('axes', titlesize=FSIZE_SMALL)  # fontsize of the axes title
plt.rc('axes', labelsize=FSIZE_SMALL)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=FSIZE_SMALL)  # fontsize of the tick labels
plt.rc('ytick', labelsize=FSIZE_SMALL)  # fontsize of the tick labels
plt.rc('legend', fontsize=FSIZE_SMALL)  # legend fontsize
plt.rc('figure', titlesize=FSIZE_LARGE)  # fontsize of the figure title
plt.rc('axes', linewidth=0.5)  # fontsize of the figure title

prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
prop_cycle_colors = prop_cycle_colors[:3] + prop_cycle_colors[4:]
region_colors = {}
for idx, r in enumerate(['US.NJ', 'US.NY', 'CAN', 'MEX', 'CHN', 'DEU', 'EUR']):
    region_colors[r] = prop_cycle_colors[idx]


def plot_consumption_time_series(_data, _clip, _t0=0, _outfile=None, _show=True, _csv_outfile=None,
                                 _output_resolution=None):
    scale_factor = 1.0
    fig_width = MAX_FIG_WIDTH_NARROW * scale_factor
    fig_height = 5
    us_states_linewidth = 0.5
    regions_linewidth = 1
    fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(fig_width, fig_height))
    for ax in axs.flatten():
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    us_state_list = [r for r in _data.get_regions() if r[:3] == 'US.' and r not in ['US.NJ', 'US.NY']]
    greys_map = plt.cm.get_cmap('Greys')
    csv_output = pd.DataFrame(index=pd.MultiIndex.from_product([us_state_list + ['US.NJ', 'US.NY', 'MEX', 'CAN', 'EUR', 'DEU', 'CHN'],
                                                                np.arange(_clip)])
                              )
    for col_idx, var in enumerate(['consumption_price', 'consumption']):
        greys = [greys_map(0.2 + (0.7 * i) / len(us_state_list)) for i in range(len(us_state_list))]
        for state in us_state_list:
            color = greys[-1]
            baseline = _data.get_regions(state).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()[0]
            time_series = _data.get_regions(state).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()
            time_series = (time_series / baseline - 1) * 100
            axs[0, col_idx].plot(time_series[_t0:], color=color, linewidth=us_states_linewidth)
            greys = greys[:-1]
            csv_output.loc[state, var] = time_series
    region_lists = [['US.NJ', 'US.NY'], ['MEX', 'CAN'], ['EUR', 'DEU', 'CHN']]
    for col_idx, var in enumerate(['consumption_price', 'consumption']):
        for row_idx, region_list in enumerate(region_lists):
            axs[row_idx, col_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            for region in region_list:
                color = region_colors[region]
                label = region
                if region in ['US.NJ', 'US.NY']:
                    label = region[3:]
                baseline = _data.get_regions(region).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()[
                    0]
                time_series = _data.get_regions(region).get_vars(var).get_sectors('FCON').clip(
                    _clip).get_data().flatten()
                time_series = (time_series / baseline - 1) * 100
                axs[row_idx, col_idx].plot(time_series[_t0:], color=color, linewidth=regions_linewidth, label=label)
                csv_output.loc[region, var] = time_series
            loc = 'lower right'
            if row_idx == 2:
                loc = 'upper right'
            axs[row_idx, 1].legend(frameon=False, loc=loc, labelspacing=0.25, handlelength=1.5, borderpad=0.2)
    for ax in axs.flatten():
        ax.set_xticks([25 * i for i in range(int(_clip / 25) + 1)])
    for ax in axs[:2, :].flatten():
        ax.set_xticklabels([])
    plt.tight_layout()
    for idx, ax in enumerate(axs.flatten()):
        pos_old = ax.get_position()
        x_new = pos_old.x0
        y_new = pos_old.y0
        width_new = pos_old.width * 1.105
        height_new = pos_old.height * 0.95
        if idx in [2, 3]:
            y_new = pos_old.y0 #+ 0.015
        if idx in [4, 5]:
            y_new = pos_old.y0 #+ 2 * 0.015
        if idx % 2 == 0:
            x_new = 0.14
        else:
            x_new = 1 - width_new
        if idx in [3]:
            ax.set_yticklabels(ax.get_yticklabels()[:-2] + [''])
        # if idx in [0]:
        #     ax.set_yticklabels(ax.get_yticklabels()[:-1] + [''])
        if idx == 4:
            ax.set_ylim(ax.get_ylim()[0], 0.01)
        if idx == 5:
            ax.set_ylim(ax.get_ylim()[0], 0.035)
        ax.set_position([x_new, y_new, width_new, height_new])
        fig.text(x_new - 0.1, ax.get_position().y0 + ax.get_position().height, chr(idx + 97),
                 horizontalalignment='left',
                 verticalalignment='center', fontweight='bold')
    center_x_left = axs[0, 0].get_position().x0 + axs[0, 0].get_position().width / 2
    center_x_right = axs[0, 1].get_position().x0 + axs[0, 1].get_position().width / 2
    # fig.text(center_x_left, 1, 'Consumption price\n(% baseline)', horizontalalignment='center', verticalalignment='top')
    # fig.text(center_x_right, 1, 'Consumption\n(% baseline)', horizontalalignment='center', verticalalignment='top')
    fig.text(center_x_left, 1, 'Consumption price', horizontalalignment='center', verticalalignment='top')
    fig.text(center_x_right, 1, 'Consumption', horizontalalignment='center', verticalalignment='top')
    xlabel_y = 0
    fig.text(center_x_left, xlabel_y, 'days', horizontalalignment='center', verticalalignment='bottom')
    fig.text(center_x_right, xlabel_y, 'days', horizontalalignment='center', verticalalignment='bottom')
    fig.text(0, 0.5, 'changes from baseline [%]', ha='left', va='center', rotation=90)
    if _outfile is not None:
        fig.savefig(_outfile, dpi=300 if _output_resolution is None else _output_resolution)
    if isinstance(_csv_outfile, str):
        csv_output = csv_output.reset_index().rename({'level_0': 'region', 'level_1': 'day'}, axis=1)
        csv_output.to_csv(_csv_outfile)
    if _show:
        plt.show()


def plot_consumption_time_series_param_variation(_clip, _t0=0, _outfile=None, _show=True, _t_variation=None,
                                                 _s_variation=None, _csv_outfile=None, _output_resolution=None):
    if _t_variation is None and _s_variation is not None:
        _t_variation = [60] * len(_s_variation)
        variation = 's'
    elif _t_variation is not None and _s_variation is None:
        _s_variation = [1.0] * len(_t_variation)
        variation = 't'
    else:
        raise ValueError('Exactly one of _t_variation and _s_variation must be given.')
    scale_factor = 1.0
    fig_width = MAX_FIG_WIDTH_NARROW * scale_factor
    fig_height = 6.5
    regions_linewidth = 1
    region_list = ['US.NJ', 'US.NY', 'MEX', 'CAN', 'EUR', 'DEU', 'CHN']
    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(fig_width, fig_height))
    for idx, ax in enumerate(axs.flatten()[:-(len(axs.flatten()) - len(region_list))]):
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    var = 'consumption'
    if variation == 't':
        csv_output = pd.DataFrame(
            index=pd.MultiIndex.from_product([region_list, _t_variation, [1.0], np.arange(_clip)]),
            columns=[var]
        )
    elif variation == 's':
        csv_output = pd.DataFrame(
            index=pd.MultiIndex.from_product([region_list, [60],  _s_variation, np.arange(_clip)]),
            columns=[var]
        )
    for idx, r in enumerate(region_list):
        ax = axs.flatten()[idx]
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        color = region_colors[r]
        for count, (t, s) in enumerate(zip(_t_variation, _s_variation)):
            filepath = "../data/acclimate_output/2021_05_21_bi_variation/bi_t{}_s{}/bi_t{}_s{}__data.pk".format(t, s, t,
                                                                                                                s)
            d = pickle.load(open(filepath, 'rb'))
            baseline = d.get_regions(r).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()[0]
            time_series = d.get_regions(r).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()
            time_series = (time_series / baseline - 1) * 100
            ax.plot(time_series[_t0:], color=color, linewidth=regions_linewidth, alpha=1 - count * 0.2)
            csv_output.loc[(r, t, s)] = time_series.reshape(-1, 1)
        ax.text(0.95, 0.05, r, transform=ax.transAxes, ha='right', va='bottom')
    for ax in axs.flatten():
        ax.set_xticks([25 * i for i in range(int(_clip / 25) + 1)])
    for ax in axs.flatten()[-(len(axs.flatten()) - len(region_list)):]:
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_yticks([])
        ax.set_xticks([])
    for count, (t, s) in enumerate(zip(_t_variation, _s_variation)):
        x = 0
        y = 0.9 - count * 0.2
        ax = axs.flatten()[-1]
        if variation == 's':
            text = '{0:3.0f}'.format(s * 100) + "% intensity"
        elif variation == 't':
            text = str(t) + " days"
        ax.plot([x, x + 0.15], [y, y], c='k', alpha=1 - count * 0.2, linewidth=1, transform=ax.transAxes)
        ax.text(x + 0.2, y, text, va='center', ha='left', transform=ax.transAxes)
    plt.tight_layout()
    for idx, ax in enumerate(axs.flatten()):
        x_old, y_old, width_old, height_old = ax.get_position().x0, ax.get_position().y0, ax.get_position().width, ax.get_position().height
        ax.set_position((x_old + 0.02 * (2 - idx % 2), y_old, width_old - 0.03, height_old))
    fig.text(0, 0.5, 'consumption changes from baseline [%]', ha='left', va='center', rotation=90)
    for idx, ax in enumerate(axs.flatten()[:-(len(axs.flatten()) - len(region_list))]):
        fig.text(0, 1, chr(idx + 97), fontweight='bold', ha='right', va='bottom', transform=ax.transAxes)
    if _outfile is not None:
        fig.savefig(_outfile, dpi=300 if _output_resolution is None else _output_resolution)
    if isinstance(_csv_outfile, str):
        csv_output = csv_output.reset_index().rename({'level_0': 'region',
                                                      'level_1': 'forcing_duration',
                                                      'level_2': 'initial_intensity_factor',
                                                      'level_3': 'day'}, axis=1)
        csv_output.to_csv(_csv_outfile)
    if _show:
        plt.show()


def plot_agent_time_series(_data: AggrData, _clip, _t0=0, _outfile=None, _show=True, _csv_outfile=None,
                           _output_resolution=None):
    scale_factor = 0.7
    fig_width = MAX_FIG_WIDTH_WIDE * scale_factor
    fig_height = 6.5

    vars_to_plot = {
        'production_price': 'Production price',
        'demand_exceedence': 'Demand exceedance',
        'effective_production_capacity': 'Capacity utilization'
    }
    regions_to_plot = {
        'US.NJ': 'NJ',
        'US.NY': 'NY',
        'USA_REST_SANDY': 'USA-OTH'
    }
    _data = copy.deepcopy(
        _data.get_regions([i for i in _data.get_regions() if i[:3] == 'US.'] + ['USA_REST_SANDY']).clip(_clip))
    _data.drop_var('production_price', _inplace=True)
    _data.calc_prices(_inplace=True)
    _data.calc_eff_forcing(_inplace=True)
    _data.calc_eff_prod_capacity(_inplace=True)
    _data.calc_demand_exceedence(_inplace=True)

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(fig_width, fig_height))
    for ax_idx, ax in enumerate(axs):
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.axhline(y=0, linewidth=1, color='k', linestyle='--')
        # ax.set_ylabel(list(vars_to_plot.values())[ax_idx])
    axs[-1].set_xlabel('days')

    csv_output_df = pd.DataFrame(columns=['region', 'day', 'production_price_change', 'demand_exceedance', 'capacity_utilization_change'])
    for region in regions_to_plot:
        region_df = pd.DataFrame(columns=csv_output_df.columns)
        for ax_idx, var in enumerate(vars_to_plot):
            ts = _data.get_vars(var).get_regions(region).get_sectors('PRIVSECTORS').get_data().flatten()
            if var in ['effective_production_capacity', 'production_price']:
                ts = (ts / ts[0]) - 1
            axs[ax_idx].plot(ts[_t0:] * 100, linewidth=1.5, label=regions_to_plot[region])
            # axs[ax_idx].plot(ts[_t0:] * 100, color=region_colors[region], linewidth=1.5, label=regions_to_plot[region])
            region_df[region_df.columns[ax_idx + 2]] = ts
            if ax_idx == 0:
                region_df['region'] = [regions_to_plot[region]] * len(ts)
                region_df['day'] = np.arange(len(ts))
        csv_output_df = csv_output_df.append(region_df, ignore_index=True, verify_integrity=True)
    axs[0].legend(frameon=False, loc='upper left', labelspacing=0.25, bbox_to_anchor=(1.01, 1.01))

    plt.tight_layout()

    # axs[1].set_ylim((axs[1].get_ylim()[0], 225))
    # axs[2].set_ylim((-7, axs[2].get_ylim()[1]))

    for ax_idx, label in enumerate(list(vars_to_plot.values())):
        axis_pos = axs[ax_idx].get_position()
        axs[ax_idx].set_position([axis_pos.x0 + 0.04, axis_pos.y0, axis_pos.width - 0.01, axis_pos.height])
        y_pos = axs[ax_idx].get_position().y0 + axs[ax_idx].get_position().height / 2
        fig.text(0, y_pos, label, verticalalignment='center', fontsize=FSIZE_SMALL, rotation=90)
        fig.text(0, y_pos, "\n(% baseline)", verticalalignment='center', fontsize=FSIZE_SMALL, rotation=90)
        fig.text(0, axs[ax_idx].get_position().y0 + axs[ax_idx].get_position().height, chr(ax_idx + 97),
                 horizontalalignment='left', verticalalignment='top', fontweight='bold')

    if _outfile is not None:
        fig.savefig(_outfile, dpi=300 if _output_resolution is None else _output_resolution)
    if isinstance(_csv_outfile, str):
        csv_output_df.to_csv(_csv_outfile)
    if _show:
        plt.show()


def plot_schematic_time_series(_outfile=None, _show=True, _output_resolution=None):
    _t0 = 4
    _data = pickle.load(open("../data/acclimate_output/sandy_data_first_draft__data.pk", "rb"))
    scale_factor = 1.0
    fig_width = MAX_FIG_WIDTH_NARROW * scale_factor
    fig_height = 6.5
    _len = 180
    _data = copy.deepcopy(_data.get_regions('US.NJ'))
    _data.drop_var('production_price', _inplace=True)
    _data.calc_prices(_inplace=True)
    _data.calc_eff_forcing(_inplace=True)
    _data.calc_eff_prod_capacity(_inplace=True)
    _data.calc_demand_exceedence(_inplace=True)
    variables = ['forcing', 'production_price', 'demand_exceedence', 'effective_production_capacity']
    variable_names = ['Production capacity\nreduction factor', 'Production\nprice', 'Demand\nexceedance', 'Capacity\nutilization']
    shading_areas = {
        'production_price': {
            'x': [0, 141 - _t0],
            'y': [],
            'labels': ['price pressure\ndevelopment'],
            'rotation': [0],
            'y_pos': [0.65],
        },
        'demand_exceedence': {
            'x': [0, 141 - _t0],
            'y': [],
            'labels': ['exponential recovery,\nnoisy demand'],
            'rotation': [0],
            'y_pos': [0.5],
        },
        'effective_production_capacity': {
            'x': [0, 114 - _t0, 141 - _t0],
            'y': [],
            'labels': ['demand\nnot satisfied', 'demand\nsatisfied'],
            'rotation': [0, 0],
            'y_pos': [0.5, 0.5],
        },
    }
    annotation_points = {
        'production_price': {
            'xy': [(143 - _t0, -0.047665)],
            'labels': ['price drop'],
            'xytext': [(160, -0.03)],
        },
        'demand_exceedence': {
            'xy': [(141 - _t0, -0.17), (142 - _t0, 0.08)],
            'labels': ['demand drop', 'demand overshoot'],
            'xytext': [(110, -0.45), (150, 0.4)],
        },
        'effective_production_capacity': {
            'xy': [(139 - _t0, 0), (142 - _t0, -0.1153)],
            'labels': ['production regime\nchange', 'production drop'],
            'xytext': [(152, 0.02), (150, -0.145)],
        },
    }
    y_lims = {
        'production_price': (-0.06, 0.07),
        'demand_exceedence': (-0.5, 2.2),
        'effective_production_capacity': (-0.15, 0.05),
        'forcing': None,
    }
    fig, axs = plt.subplots(4, 1, sharex=True, sharey=False, figsize=(fig_width, fig_height))
    for ax_idx, ax in enumerate(axs):
        ax.axhline(y=0 if ax_idx > 0 else 1, color='k', linestyle='--', linewidth=1)
        var = variables[ax_idx]
        _clip = _len
        if var in ['production_price']:
            _clip = 145
        elif var in ['demand_exceedence', 'effective_production_capacity']:
            _clip = 143
        ts = _data.get_vars(var).get_regions('US.NJ').get_sectors('MACH').get_data().flatten()[:_clip]
        if var == 'demand_exceedence':
            ts[5] = 1.8
        elif var == 'effective_production_capacity':
            ts[142] = 0.925
        if var in ['effective_production_capacity', 'production_price']:
            ts = (ts / ts[0]) - 1
        fill_val = 0
        if var == 'forcing':
            fill_val = 1
        ax.plot(list(ts[_t0:]) + [fill_val] * (_len - _clip), color='k', linewidth=1)
        ax.set_ylim(y_lims[var])
        if var in shading_areas.keys():
            for area_idx, area_label in enumerate(shading_areas[var]['labels']):
                ax.axvspan(shading_areas[var]['x'][area_idx], shading_areas[var]['x'][area_idx + 1],
                           alpha=0.1 + 0.1 * area_idx % 2, color='k', linewidth=0)
                ax.text(shading_areas[var]['x'][area_idx] + (
                        shading_areas[var]['x'][area_idx + 1] - shading_areas[var]['x'][area_idx]) / 2,
                        ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * shading_areas[var]['y_pos'][area_idx],
                        area_label, rotation=shading_areas[var]['rotation'][area_idx],
                        horizontalalignment='center', verticalalignment='center', fontsize=FSIZE_TINY)
        if var in annotation_points.keys():
            for annotation_idx, annotation_label in enumerate(annotation_points[var]['labels']):
                xy = annotation_points[var]['xy'][annotation_idx]
                xytext = annotation_points[var]['xytext'][annotation_idx]
                ax.annotate(annotation_label, xy, xytext=xytext, arrowprops={'arrowstyle': '->'},
                            ha='center', fontsize=FSIZE_TINY)
    for ax in axs.flatten():
        ax.set_xticks([25 * i for i in range(int(_len / 25) + 1)])
    for ax_idx, ax in enumerate(axs.flatten()):
        # ax.set_xticks([0])
        ax.set_xticklabels([])
        ax.set_yticks([0])
        ax.set_yticklabels(['0'])
        if ax_idx == 0:
            lambda_0 = _data.get_vars('forcing').get_sectors('MACH').get_data().min()
            ax.set_yticks([0, lambda_0, 1])
            ax.set_yticklabels(['0', r'$\lambda_0$', '1'])
            print(lambda_0)
    plt.tight_layout()
    for idx, ax in enumerate(axs.flatten()):
        pos_old = ax.get_position()
        x_new = 0.15
        y_new = pos_old.y0 + 0.035 * (idx + 1)
        width_new = 1 - x_new
        height_new = pos_old.height * 0.88
        ax.set_position([x_new, y_new, width_new, height_new])
        fig.text(0, ax.get_position().y0 + ax.get_position().height, chr(idx + 97), ha='left', va='center',
                 fontweight='bold')
    for ax_idx, ax in enumerate(axs):
        title_y = ax.get_position().y0 + axs[0].get_position().height / 2
        for line_nr, text in enumerate(variable_names[ax_idx].split('\n')):
            fig.text(0 + line_nr * 0.04, title_y, text, ha='left', va='center', rotation=90)
    time_ax = fig.add_subplot()
    time_ax_pos = [0.1, 0.03, 0.9, 0.08]
    time_ax.set_position(time_ax_pos)
    time_ax.set_ylim([0, 1])
    time_ax.set_xlim(axs[-1].get_xlim())
    time_ax.spines['right'].set_visible(False)
    time_ax.spines['top'].set_visible(False)
    time_ax.spines['left'].set_visible(False)
    time_ax.spines['bottom'].set_position(('data', 0.25))
    time_ax.set_yticks([])
    time_ax.set_xticks([])
    time_ax.set_xticks([25 * i for i in range(int(_len / 25) + 1)])
    time_ax.set_xticklabels(['0'] + (len(time_ax.get_xticks()) - 1) * [''])
    time_ax.set_yticks([])
    time_ax.set_yticklabels(len(time_ax.get_yticks()) * [''])
    time_ax.axvline(x=time_ax.get_xlim()[0], ymin=0, ymax=0.5, color='k', linewidth=plt.rcParams['axes.linewidth'])
    fig.text(0, time_ax_pos[1] + time_ax_pos[3] * 0.25, 'Timeline', horizontalalignment='left',
             verticalalignment='center', rotation=90)
    fig.text(0, time_ax_pos[1] + time_ax_pos[3], chr(4 + 97), horizontalalignment='left',
             verticalalignment='center', fontweight='bold')
    timeline_annotations = {
        'disaster': {
            'xy': (0, 0.25),
            'xytext': (0, 0.75),
        },
        'consumption\nprice peak': {
            'xy': (141 - _t0, 0.25),
            'xytext': (141 - _t0, 1),
        }
    }
    for timeline_event_name, timeline_event_params in timeline_annotations.items():
        time_ax.annotate(timeline_event_name, timeline_event_params['xy'], xytext=timeline_event_params['xytext'],
                         arrowprops={'arrowstyle': '->'}, horizontalalignment='center', fontsize=FSIZE_TINY)
    time_ax.annotate('upstream and downstream effects', xy=((141 - _t0) / 2, 0.4), xytext=((141 - _t0) / 2, 0.75),
                     xycoords='data',
                     ha='center', va='bottom',
                     arrowprops=dict(arrowstyle='-[, widthB={}, lengthB={}'.format(12.8, 0.5), lw=1.0),
                     fontsize=FSIZE_TINY)
    time_ax.annotate('normalization', xy=((141 - _t0) + (time_ax.get_xlim()[1] - (141 - _t0)) / 2, 0.4),
                     xytext=((141 - _t0) + (time_ax.get_xlim()[1] - (141 - _t0)) / 2, 0.75),
                     xycoords='data',
                     ha='center', va='bottom',
                     arrowprops=dict(arrowstyle='-[, widthB={}, lengthB={}'.format(4, 0.5), lw=1.0),
                     fontsize=FSIZE_TINY)
    center_x = axs[-1].get_position().x0 + axs[0].get_position().width / 2
    fig.text(center_x, 0, 'time', horizontalalignment='center', verticalalignment='bottom')
    if _outfile is not None:
        fig.savefig(_outfile, dpi=300 if _output_resolution is None else _output_resolution)
    if _show:
        plt.show()


def plot_consumption_deviation_map(_data, _clip, _t0, _outfile=None, _excluded_countries=None, _only_usa=False,
                                   _numbering=None, _csv_outfile=None, _var='consumption_deviation', _show_cbar=True,
                                   _v_limits=None, _output_resolution=None):
    scale_factor = 0.75
    var_labels = {
        'consumption_deviation': 'Consumption change (%)',
        'consumption_loss': 'Consumption loss (USD)'
    }
    if _excluded_countries is None:
        _excluded_countries = []
    _data = _data.get_sectors('FCON').get_vars('consumption')
    consumption = _data.get_data()[..., _t0:_clip]
    consumption_loss = consumption.sum(axis=-1).flatten() - consumption[..., 0].flatten() * (_clip - _t0)
    consumption_deviation = consumption_loss / (consumption[..., 0].flatten() * (_clip - _t0))
    consumption_deviation = consumption_deviation * 100
    regions = _data.get_regions()
    df = pd.DataFrame()
    df['region'] = regions
    df['consumption_deviation'] = consumption_deviation
    df['consumption_loss'] = consumption_loss
    df.set_index('region', drop=True)
    df.loc[_excluded_countries, ['consumption_deviation', 'consumption_loss']] = np.nan
    df = df.loc[df.index.intersection(list(WORLD_REGIONS['WORLD']))]
    if _only_usa:
        df_reduced = df.loc[df.index.intersection(list(set(WORLD_REGIONS['WORLD']) - {'USA', 'CHN'}))]
        df_reduced = df_reduced[df_reduced['region'].apply(lambda x: True if x[:3] in ['USA', 'US.'] else False)]
    else:
        df_reduced = df.loc[
            df.index.intersection(list(set(WORLD_REGIONS['WORLD']) - {'CHN'} - set(WORLD_REGIONS['USA'])) + ['USA'])]
    if _v_limits is None:
        min_val = df_reduced[_var].min()
        max_val = df_reduced[_var].max()
        _v_limits = (min_val, max_val)
    else:
        (min_val, max_val) = _v_limits
    if _only_usa:
        c1, c2 = 'purple', 'orange'
        if max_val > 0 > min_val:
            x_white = abs(min_val) / (abs(min_val) + max_val)
            min_alpha = abs(min_val) / max(abs(min_val), max_val)
            max_alpha = max_val / max(abs(min_val), max_val)
            cm = create_colormap('custom',
                                 [c1, 'white', c2],
                                 alphas=[min_alpha, 0, max_alpha],
                                 xs=[0, x_white, 1]
                                 )
        elif df_reduced[_var].max() < 0:
            cm = create_colormap('custom',
                                 [c1, 'white'],
                                 alphas=[1, 0],
                                 xs=[0, 1]
                                 )
        else:
            cm = create_colormap('custom',
                                 ['white', c2],
                                 alphas=[0, 1],
                                 xs=[0, 1]
                                 )
    else:
        c1, c2, c3 = 'm', '#f64748', 'royalblue'
        if max_val > 0 > min_val:
            df_no_us = df_reduced.loc[list(set(df_reduced.index) - {'USA'})]
            intermediate_min_val = df_no_us[_var].min()
            x_white = abs(min_val) / (abs(min_val) + max_val)
            x_red = (intermediate_min_val - min_val) / (max_val - min_val)
            min_alpha = 1
            intermediate_alpha = abs(intermediate_min_val) / max(abs(intermediate_min_val), max_val)
            max_alpha = max_val / max(abs(intermediate_min_val), max_val)
            cm = create_colormap('custom',
                                 [c1, c2, "white", c3],
                                 alphas=[min_alpha, intermediate_alpha, 0, max_alpha],
                                 xs=[0, x_red, x_white, 1]
                                 )
    fig_width = MAX_FIG_WIDTH_WIDE * scale_factor * 0.5
    if not _show_cbar:
        fig_width = fig_width * 0.8
    fig_height = MAX_FIG_WIDTH_WIDE * 0.4 * scale_factor * 0.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 0.03 if _show_cbar else 0])
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1]) if _show_cbar else None
    if _only_usa:
        patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/usa_new.pkl.gz"
        lims = (-13, 15, -34, 25)
    else:
        patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/map_robinson_0.1simplified.pkl.gz"
        lims = None
    make_map(patchespickle_file=patchespickle_file,
             regions=df_reduced['region'],
             data=df_reduced[_var],
             y_ticks=None,
             y_label=var_labels[_var],
             numbering=None,
             numbering_fontsize=FSIZE_TINY,
             extend_c="both",
             ax=ax,
             cax=cax,
             cm=cm,
             y_label_fontsize=FSIZE_TINY,
             y_ticks_fontsize=FSIZE_TINY,
             ignore_regions=_excluded_countries,
             lims=lims,
             only_usa=_only_usa,
             v_limits=_v_limits,
             show_cbar=_show_cbar,
             )
    if not _only_usa:
        ax.set_position((0, 0, 0.78, 1))
        cax.set_position((0.79, 0.02, 0.02, 0.96))
    else:
        ax.set_position((0.04, 0, 0.78, 1))
        cax.set_position((0.81, 0.02, 0.02, 0.96))
    if _numbering is not None:
        x_pos = 0
        if _only_usa:
            x_pos = 0.05
        fig.text(x_pos, 1, _numbering, fontweight='bold', ha='left', va='top', fontsize=FSIZE_TINY)
    if isinstance(_outfile, str):
        fig.savefig(_outfile, dpi=300 if _output_resolution is None else _output_resolution)
    if isinstance(_csv_outfile, str):
        df_reduced.sort_values(by='region').to_csv(_csv_outfile)
    plt.show(block=False)
    return df


def make_import_dependency_figure_for_paper(_data: AggrData, _clip=None, _t0=None, _outfile=None,
                                            _output_resolution=None):
    scale_factor = 0.8

    regions_to = ['MEX', 'CAN', 'EUR', 'CHN']

    fig, axs = plt.subplots(1, 2, figsize=(MAX_FIG_WIDTH_WIDE * scale_factor, 2.5), sharex=True)

    if _clip is not None:
        _data = _data.clip(_clip)
    if _t0 is not None:
        _data = _data.clip(_t0, _data.get_sim_duration())

    data_sel_regions = _data.get_regions(regions_to).get_sectors('FCON').get_vars('consumption')
    consumption_losses = data_sel_regions.get_data().sum(axis=-1).flatten() - data_sel_regions.get_data()[
        ..., 0].flatten() * _data.get_sim_duration()
    regions_world = list(set(WORLD_REGIONS['WORLD']) - set(list(WORLD_REGIONS.keys())))
    data_world = _data.get_regions(regions_world).get_sectors('FCON').get_vars('consumption')
    consumption_row = (data_world.get_data().sum(axis=1) - data_sel_regions.get_data().sum(axis=1)).flatten()
    data_usa = _data.get_regions('USA').get_sectors('FCON').get_vars('consumption')
    consumption_row = consumption_row - data_usa.get_data().flatten()
    consumption_losses_row = consumption_row.sum() - consumption_row[0] * _data.get_sim_duration()
    consumption_losses = consumption_losses / (data_sel_regions.get_data()[..., 0].flatten() * _data.get_sim_duration())
    consumption_losses_row = consumption_losses_row / (consumption_row[0] * _data.get_sim_duration())
    regions_to = regions_to + ['ROW']
    consumption_losses = np.concatenate([consumption_losses, [consumption_losses_row]])
    consumption_losses = consumption_losses * 100
    axs[0].axhline(y=0, linestyle='--', color='k', linewidth=1)
    axs[0].bar(np.arange(len(regions_to)), consumption_losses, color='firebrick')
    y_vals = axs[0].get_yticks()
    axs[0].set_ylabel('Consumption deviation\n(% baseline)')
    axs[0].set_xticks(range(len(regions_to)))
    axs[0].set_xticklabels(regions_to)
    axs[0].set_xlabel('Region')

    balances = pd.read_csv("../data/generated/us_import_balances.csv", index_col=0)
    balances['flow_value'] = balances['flow_value'].apply(lambda x: '0.0' if x == '--' else x)
    balances['rel_to_all_imports'] = balances['rel_to_all_imports'].apply(lambda x: '0.0' if x == '--' else x)
    balances['flow_value'] = balances['flow_value'].astype(float)
    balances['rel_to_all_imports'] = balances['rel_to_all_imports'].astype(float)

    width = 0.35
    import_shares = [balances[balances['region_to'] == reg]['rel_to_all_imports'].item() * 100 for reg in regions_to]
    export_shares = [balances[balances['region_from'] == reg]['rel_to_all_exports'].item() * 100 for reg in regions_to]
    x = np.arange(len(regions_to))
    rects1 = axs[1].bar(x - width / 2, import_shares, width, label='Import')
    rects2 = axs[1].bar(x + width / 2, export_shares, width, label='Export')
    axs[1].set_ylabel('US trade share\n(% of total import / export)')
    axs[1].set_xlabel('Region')
    axs[1].legend(frameon=False, loc='upper right', labelspacing=0.25)

    fig.tight_layout()

    axs[0].set_position([axs[0].get_position().x0, axs[0].get_position().y0, axs[0].get_position().width - 0.025,
                         axs[0].get_position().height])
    axs[1].set_position(
        [axs[1].get_position().x0 + 0.025, axs[1].get_position().y0, axs[1].get_position().width - 0.025,
         axs[1].get_position().height])

    for ax_idx, ax in enumerate(axs):
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax_pos = ax.get_position()
        fig.text(ax_pos.x0 - ax_pos.width * 0.325, ax_pos.y0 + ax_pos.height, chr(ax_idx + 97),
                 horizontalalignment='right',
                 verticalalignment='top', fontweight='bold')
    if isinstance(_outfile, str):
        fig.savefig(_outfile, dpi=300 if _output_resolution is None else _output_resolution)
    plt.show()


def plot_import_dependency_scatter(_data: AggrData, _clip=None, _t_0=0, x_var='export_to_US',
                                   y_var='difference', z_var='relative_loss',
                                   _x_scale='symlog', _y_scale='symlog', _acclimate_var='consumption',
                                   _acclimate_sector='FCON', _regression=False, _residualsplot=False, _outfile=None,
                                   _numbering=None, _show_xlabel=True, _show_ylabel=True, _show_xticklabels=True,
                                   _excluded_countries=None, _legend=False, _show=True, _xlim=None, _ylim=None,
                                   _s_offset=None, _s_scale=None, _csv_outfile=None, _output_resolution=None,
                                   print_lims=False):
    var_data = _data.get_vars(_acclimate_var).get_sectors(_acclimate_sector)
    if _excluded_countries is not None:
        var_data = var_data.get_regions(list(set(var_data.get_regions()) - set(_excluded_countries)))
    var_data.data_capsule.data = var_data.get_data() * 1000  # convert to single dollars
    var_diff = copy.deepcopy(var_data.clip(1))
    var_diff_data = var_data.clip(_clip).get_data().sum(axis=-1, keepdims=True) - var_data.clip(1).get_data() * _clip
    var_diff.data_capsule.data = var_diff_data
    var_dev = copy.deepcopy(var_data.clip(1))
    var_dev_data = (var_data.clip(_clip).get_data().sum(axis=-1, keepdims=True) / (
            var_data.clip(1).get_data() * _clip) - 1) * 100
    var_dev.data_capsule.data = var_dev_data
    var_max = copy.deepcopy(var_data.clip(1))
    var_max.data_capsule.data = var_data.clip(55).get_data().max(axis=-1, keepdims=True)
    trade_flows_path = "../data/generated/us_trade_flows.csv"
    if os.path.exists(trade_flows_path):
        trade_flows = pd.read_csv(trade_flows_path, index_col=0)
    else:
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
        trade_flows['us_trade_volume'] = trade_flows['export_to_US'] + trade_flows['import_from_US']
        trade_flows.to_csv(trade_flows_path)
    dataframe = pd.DataFrame(columns=['region', 'difference', 'absolute_difference',
                                      'relative_loss', 'maximum'])
    dataframe.set_index('region', drop=True)
    for r in var_data.get_regions():
        dataframe.loc[r] = [r, var_diff.get_regions(r).get_data().flatten()[0],
                            abs(var_diff.get_regions(r).get_data().flatten()[0]),
                            var_dev.get_regions(r).get_data().flatten()[0],
                            var_max.get_regions(r).get_data().flatten()[0]]
    plot_data = trade_flows.join(dataframe, how='inner')
    print(plot_data[z_var].max(), plot_data[z_var].min())
    print(plot_data.loc['CHN'])
    for variable, scale in zip([x_var, y_var], [_x_scale, _y_scale]):
        if scale in ['symlog', 'log']:
            zero_rows = plot_data[plot_data[variable] == 0].index
            if len(zero_rows) > 0:
                print('Dropping {}'.format(zero_rows))
                plot_data.drop(zero_rows, inplace=True)
    plot_data.sort_values(by=z_var, ascending=False, inplace=True)
    label_names = {
        'difference': '{} change'.format(_acclimate_var) + '\n({}USD)',
        'absolute_difference': 'Absolute {} change'.format(_acclimate_var) + '\n({}USD)',
        'relative_loss': 'Relative {} loss (%)'.format(_acclimate_var),
        'maximum': '{} maximum'.format(_acclimate_var) + '\n({}USD)',
        'us_import_share': 'US import share (%){}',
        'us_export_share': 'US export share (%){}',
        'us_trade_balance': 'US trade balance ({}USD)',
        'us_trade_volume': 'US trade volume ({}USD)',
        'total_export': 'Total export ({}USD)',
        'total_import': 'Total import ({}USD)',
        'export_to_US': 'Export to the US ({}USD)',
        'import_from_US': 'Import from the US ({}USD)',
    }
    for key, scale in zip([x_var, y_var], [_x_scale, _y_scale]):
        label_names[key] = label_names[key].format(
            'millions of ' if scale == 'linear' and key not in ['us_import_share', 'us_export_share',
                                                                'relative_loss'] else '')
    height_width_ratio = 0.93
    fig_width = MAX_FIG_WIDTH_NARROW
    fig_height = MAX_FIG_WIDTH_NARROW * height_width_ratio
    ax_width, ax_height = 0.82, 0.82 / height_width_ratio
    if not _show_ylabel:
        fig_width = MAX_FIG_WIDTH_NARROW * 0.9
        ax_width = ax_width / 0.9
    if not _show_xlabel:
        fig_height = MAX_FIG_WIDTH_NARROW * height_width_ratio * 0.95
        ax_height = ax_height / 0.95
    x_0, y_0, = 1 - ax_width, 1 - ax_height
    figsize = (fig_width, fig_height)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_position([x_0, y_0, ax_width, ax_height])
    ax.set_yscale(_y_scale)
    ax.set_xscale(_x_scale)
    # min_flow = 1e9 / 365 # EORA data is per days (probably?)
    # min_flow = -1e100
    s_min, s_max = 30, 500
    if _s_offset is None or _s_scale is None:
        _s_offset = min(abs(plot_data[z_var]))
        _s_scale = s_max / (max(abs(plot_data[z_var])) - min(abs(plot_data[z_var])))
    for r in plot_data.index:
        # if plot_data.loc[r, x_var] >= min_flow:
        scatter_size = s_min + (abs(plot_data.loc[r, z_var]) - _s_offset) * _s_scale
        color = 'g' if plot_data.loc[r, z_var] > 0 else 'r'
        ax.scatter(plot_data.loc[r, x_var], plot_data.loc[r, y_var], s=scatter_size, color=color, alpha=0.3,
                   linewidths=0)
        ax.annotate(r, (plot_data.loc[r, x_var], plot_data.loc[r, y_var]), fontsize=5, ha='center')
    if _legend:
        for count, legend_entry in enumerate([0.025, 0.05, 0.1]):
            lenged_scatter_size = s_min + (legend_entry - _s_offset) * _s_scale
            x = 0.6 + count * 0.08 + 1.4 * legend_entry
            y = 0.1
            text = '{0:1.3f}'.format(legend_entry).rstrip('0') + '%'
            ax.scatter(x, y, s=lenged_scatter_size, color='grey', alpha=0.3, linewidths=0, transform=ax.transAxes)
            ax.text(x, y, text, va='center', ha='center', fontsize=5, c='k',
                    transform=ax.transAxes)
    # line_x1 = max(ax.get_ylim()[0], ax.get_xlim()[0])
    # line_x2 = line_x1 * 10
    # ax.axline((line_x1, line_x1), (line_x2, line_x2), c='k', linestyle='--')
    if _x_scale == 'linear' and x_var not in ['us_import_share', 'us_export_share', 'relative_loss']:
        ax.set_xticklabels([int(i / 1e6) for i in ax.get_xticks()])
    if _y_scale == 'linear' and y_var not in ['us_import_share', 'us_export_share', 'relative_loss']:
        ax.set_yticklabels([int(i / 1e6) for i in ax.get_yticks()])
    if _show_xlabel:
        ax.set_xlabel(label_names[x_var])
    if not _show_xticklabels:
        ax.set_xticklabels([])
    if _show_ylabel:
        ax.set_ylabel(label_names[y_var])
    else:
        ax.set_yticklabels([])
    if _xlim is not None:
        ax.set_xlim(_xlim)
    if _ylim is not None:
        ax.set_ylim(_ylim)
    if _regression:
        x, y = np.log(plot_data[x_var].values), np.log(plot_data[y_var].values)
        res = stats.linregress(x, y)
        r_2 = res.rvalue ** 2
        p_val = res.pvalue
        intercept = res.intercept
        slope = res.slope
        x1, x2 = min(plot_data[x_var]), max(plot_data[x_var])
        y1, y2 = x1 ** slope * np.e ** intercept, x2 ** slope * np.e ** intercept
        ax.axline((x1, y1), (x2, y2), c='k', linestyle='--', linewidth=1)
        ax.text(0.03, 0.95, r'$y = {0:1.3f}x^{1}$'.format(np.e ** intercept, "{" + "{0:1.3f}".format(slope) + "}"),
                ha='left', va='top', transform=ax.transAxes)
        print('R2 = {0:1.3f}\np = {1:1.3f}'.format(r_2, p_val))
    if _numbering is not None:
        transform = transforms.blended_transform_factory(fig.transFigure, ax.transAxes)
        fig.text(0 if _show_ylabel else 0.04, 1, _numbering, ha='left', va='top', transform=transform,
                 fontweight='bold')
    if _outfile is not None:
        fig.savefig(_outfile, dpi=300 if _output_resolution is None else _output_resolution)
    if isinstance(_csv_outfile, str):
        plot_data[[x_var, y_var, z_var]].to_csv(_csv_outfile)
    if print_lims:
        print(ax.get_xlim(), ax.get_ylim())
    if _show:
        plt.show()
    if _residualsplot and _regression:
        visualizer = ResidualsPlot(LinearRegression(), hist=True)
        visualizer.fit(x.reshape((-1, 1)), y)
        visualizer.show()
    return plot_data


if __name__ == '__main__':
    data = pickle.load(open("../data/acclimate_output/2021_05_21_bi_variation/bi_t60_s1.0/bi_t60_s1.0__data.pk", "rb"))
    data.calc_prices(_inplace=True)
    data.calc_eff_forcing(_inplace=True)
    data.calc_demand_exceedence(_inplace=True)
    data.calc_eff_prod_capacity(_inplace=True)
    plot_consumption_time_series(data, 134, 4, _outfile="../figures/consumption_plots.pdf",
                                 _csv_outfile="../data/publication_data/fig3.csv")
    plot_consumption_deviation_map(data, _clip=105, _t0=5, _numbering='a', _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'],
                                   _outfile="../figures/choropleth_consumption_deviation_WORLD.pdf",
                                   _csv_outfile="../data/publication_data/fig2a.csv")
    plot_consumption_deviation_map(data, _clip=105, _t0=5, _only_usa=True, _numbering='b', _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'],
                                   _outfile="../figures/choropleth_consumption_deviation_USA.pdf",
                                   _csv_outfile="../data/publication_data/fig2b.csv")
    plot_schematic_time_series(_outfile="../figures/time_series_schematic.pdf", _show=True)
    plot_agent_time_series(data, _clip=85, _t0=4, _outfile="../figures/time_series_NJ_NY_US-49.pdf",
                           _csv_outfile="../data/publication_data/fig5.csv")

    for t, numbering in zip([60, 80, 100, 120], ['a', 'b', 'c', 'd']):
        filepath = "../data/acclimate_output/2021_05_21_bi_variation/bi_t{}_s1.0/bi_t{}_s1.0__data.pk".format(t, t)
        outpath = "../figures/consumption_scatter_t_variation/consumption_scatter_t{}.pdf".format(t)
        d = pickle.load(open(filepath, 'rb'))
        plot_import_dependency_scatter(
            d, _clip=105, _t_0=5, x_var='us_trade_volume',
            y_var='absolute_difference',
            z_var='relative_loss', _x_scale='log', _y_scale='log',
            _acclimate_var='consumption', _acclimate_sector='FCON', _regression=True,
            _residualsplot=False, _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'],
            _outfile=outpath,
            _numbering=numbering,
            _show_xlabel=numbering in ['c', 'd'],
            _show_xticklabels=numbering in ['c', 'd'],
            _show_ylabel=numbering in ['a', 'c'],
            _legend=numbering == 'd',
            _s_offset=0.0001,
            _s_scale=8000,
            _csv_outfile="../data/publication_data/fig4{}.csv".format(numbering)
        )

    plot_import_dependency_scatter(data, _clip=105, _t_0=5, x_var='import_from_US', y_var='difference',
                                   z_var='relative_loss', _x_scale='linear', _y_scale='linear', _acclimate_var='consumption',
                                   _acclimate_sector='FCON', _regression=False, _residualsplot=False,
                                   _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'],
                                   _outfile="../figures/consumption_scatter_1.pdf", _numbering='a',
                                   _csv_outfile="../data/publication_data/supfig2a.csv")
    plot_import_dependency_scatter(data, _clip=105, _t_0=5, x_var='export_to_US', y_var='difference',
                                   z_var='relative_loss', _x_scale='linear', _y_scale='linear', _acclimate_var='consumption',
                                   _acclimate_sector='FCON', _regression=False, _residualsplot=False, _show_ylabel=False,
                                   _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'], _numbering='b',
                                   _outfile="../figures/consumption_scatter_2.pdf",
                                   _csv_outfile="../data/publication_data/supfig2b.csv")
    plot_import_dependency_scatter(data, _clip=105, _t_0=5, x_var='import_from_US', y_var='absolute_difference',
                                   z_var='relative_loss', _x_scale='log', _y_scale='log', _acclimate_var='consumption',
                                   _acclimate_sector='FCON', _regression=True, _residualsplot=False,
                                   _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'], _numbering='c',
                                   _outfile="../figures/consumption_scatter_3.pdf",
                                   _csv_outfile="../data/publication_data/supfig2c.csv")
    plot_import_dependency_scatter(data, _clip=105, _t_0=5, x_var='export_to_US', y_var='absolute_difference',
                                   z_var='relative_loss', _x_scale='log', _y_scale='log', _acclimate_var='consumption',
                                   _acclimate_sector='FCON', _regression=True, _residualsplot=False, _show_ylabel=False,
                                   _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'], _numbering='d',
                                   _outfile="../figures/consumption_scatter_4.pdf", _legend=True,
                                   _csv_outfile="../data/publication_data/supfig2d.csv")

    for t, numbering in zip([60, 80, 100, 120], ['b', 'd', 'f', 'h']):
        filepath = "../data/acclimate_output/2021_05_21_bi_variation/bi_t{}_s1.0/bi_t{}_s1.0__data.pk".format(t, t)
        outpath = "../figures/us_choropleth_t_variation/us_choropleth_t{}.pdf".format(t)
        d = pickle.load(open(filepath, "rb"))
        plot_consumption_deviation_map(d, _clip=105, _t0=5, _numbering=numbering,
                                       _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'],
                                       _only_usa=True, _v_limits=(-0.6, 0.1),
                                       _outfile=outpath,
                                       _csv_outfile="../data/publication_data/supfig5{}.csv".format(numbering))

    for t, numbering in zip([60, 80, 100, 120], ['a', 'c', 'e', 'g']):
        filepath = "../data/acclimate_output/2021_05_21_bi_variation/bi_t{}_s1.0/bi_t{}_s1.0__data.pk".format(t, t)
        outpath = "../figures/world_choropleth_t_variation/world_choropleth_t{}.pdf".format(t)
        d = pickle.load(open(filepath, "rb"))
        plot_consumption_deviation_map(d, _clip=105, _t0=5, _numbering=numbering,
                                       _excluded_countries=['BLR', 'SDS', 'SDN', 'MDA'],
                                       _only_usa=False, _v_limits=(-0.5, 0.07),
                                       _outfile=outpath,
                                       _csv_outfile="../data/publication_data/supfig5{}.csv".format(numbering))

    plot_consumption_time_series_param_variation(_clip=105, _t0=5, _show=False, _t_variation=[60, 80, 100, 120],
                                                 _outfile="../figures/consumption_plot_t_variation.pdf",
                                                 _csv_outfile="../data/publication_data/supfig3.csv")
    plot_consumption_time_series_param_variation(_clip=105, _t0=5, _show=False, _s_variation=[0.9, 1.0, 1.1],
                                                 _outfile="../figures/consumption_plot_s_variation.pdf",
                                                 _csv_outfile="../data/publication_data/supfig4.csv")
    pass
