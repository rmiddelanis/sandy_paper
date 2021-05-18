import copy
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from scipy.interpolate import interp1d

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


def make_consumption_figure_for_paper(_data, _clip, _t0=0, _outfile=None, _show=True):
    scale_factor = 1.0
    fig_width = MAX_FIG_WIDTH_NARROW * scale_factor
    fig_height = 5

    us_states_linewidth = 0.5
    regions_linewidth = 1

    prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prop_cycle_colors = prop_cycle_colors[:3] + prop_cycle_colors[4:]
    colors = {}
    for idx, r in enumerate(['US.NJ', 'US.NY', 'CAN', 'MEX', 'CHN', 'DEU', 'EUR']):
        colors[r] = prop_cycle_colors[idx]

    fig, axs = plt.subplots(3, 2, sharex=False, sharey=False, figsize=(fig_width, fig_height))

    for ax in axs.flatten():
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    us_state_list = [r for r in _data.get_regions() if r[:3] == 'US.' and r not in ['US.NJ', 'US.NY']]
    greys_map = plt.cm.get_cmap('Greys')
    for col_idx, var in enumerate(['consumption_price', 'consumption']):
        greys = [greys_map(0.2 + (0.7 * i) / len(us_state_list)) for i in range(len(us_state_list))]
        for state in us_state_list:
            color = greys[-1]
            baseline = _data.get_regions(state).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()[0]
            time_series = _data.get_regions(state).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()
            time_series = (time_series / baseline - 1) * 100
            axs[0, col_idx].plot(time_series[_t0:], color=color, linewidth=us_states_linewidth)
            greys = greys[:-1]

    # observed data:
    region_lists = [['US.NJ', 'US.NY'], ['MEX', 'CAN'], ['EUR', 'DEU', 'CHN']]
    for col_idx, var in enumerate(['consumption_price', 'consumption']):
        for row_idx, region_list in enumerate(region_lists):
            axs[row_idx, col_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            for region in region_list:
                color = colors[region]
                label = region
                if region in ['US.NJ', 'US.NY']:
                    label = region[3:]
                baseline = _data.get_regions(region).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()[
                    0]
                time_series = _data.get_regions(region).get_vars(var).get_sectors('FCON').clip(
                    _clip).get_data().flatten()
                time_series = (time_series / baseline - 1) * 100
                axs[row_idx, col_idx].plot(time_series[_t0:], color=color, linewidth=regions_linewidth, label=label)
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
        width_new = pos_old.width * 1.1
        height_new = pos_old.height * 0.95
        if idx in [0, 1]:
            y_new = pos_old.y0 - 0.015
        if idx in [4, 5]:
            y_new = pos_old.y0 + 0.015
        if idx % 2 == 0:
            x_new = 0.11
        else:
            x_new = 1 - width_new
        ax.set_position([x_new, y_new, width_new, height_new])
        fig.text(x_new - 0.1, ax.get_position().y0 + ax.get_position().height, chr(idx + 97),
                 horizontalalignment='left',
                 verticalalignment='center', fontweight='bold')

    center_x_left = axs[0, 0].get_position().x0 + axs[0, 0].get_position().width / 2
    center_x_right = axs[0, 1].get_position().x0 + axs[0, 1].get_position().width / 2

    # title_y = axs[0, 0].get_position().y0 + axs[0, 0].get_position().height * 1.05
    fig.text(center_x_left, 1, 'Consumption Price\n(% baseline)', horizontalalignment='center', verticalalignment='top')
    fig.text(center_x_right, 1, 'Consumption\n(% baseline)', horizontalalignment='center', verticalalignment='top')

    # xlabel_y = axs[-1, 0].get_position().y0 - 0.04
    xlabel_y = 0
    fig.text(center_x_left, xlabel_y, 'days', horizontalalignment='center', verticalalignment='bottom')
    fig.text(center_x_right, xlabel_y, 'days', horizontalalignment='center', verticalalignment='bottom')

    if _outfile is not None:
        fig.savefig(_outfile, dpi=300)

    if _show:
        plt.show()


def make_schematic_consumption_figure_for_paper(_outfile=None, _show=True):
    _len = 120
    scale_factor = 1.0
    fig_width = MAX_FIG_WIDTH_NARROW * scale_factor
    fig_height = 3

    regions_linewidth = 1

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False, figsize=(fig_width, fig_height))

    for ax in axs.flatten():
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    # schematic plots:
    pts_s = [0, 10, 50, 60, 100, 113, 125, 130]
    pts_y = [0, -0.05, 0.2, 0.2, 0.03, 0.013, 0.01, 0.01]
    f = interp1d(pts_s, pts_y, fill_value='extrapolate', kind='cubic')
    x = np.arange(_len)
    y = f(np.arange(_len))
    axs[0].plot(x, y, linewidth=regions_linewidth, color='k')
    axs[1].plot(np.arange(_len), y * -0.5, linewidth=regions_linewidth, color='k')
    event_pts_y = [0, min(y), max(y)]
    event_pts_x = [list(y).index(pt) for pt in event_pts_y]
    intersects_x = [i for i in range(1, len(y) - 1) if y[i] * y[i + 1] < 0]
    # axs[-1, 0].scatter(event_pts_x, y[event_pts_x], s=3, marker='o', c='w', edgecolors='k', linewidths=1)
    # axs[-1, 0].arrow(0, 0.1, 0, -0.1, shape='full', length_includes_head=True, color='grey', width=1, head_width=5)
    # axs[-1, 0].plot(0, 0.01, 'v', markersize=3, c='grey')
    # axs[-1, 0].plot([0, 0.01], [0, 0.03], c='grey')
    # axs[-1, 0].text(0, 0.04, 'disaster', rotation=90, fontsize=tiny_fsize, horizontalalignment='center', verticalalignment='bottom')
    for ax_idx, ax in enumerate(axs):
        ax.axvspan(0, intersects_x[0], alpha=0.1, color='k', linewidth=0)
        ax.axvspan(intersects_x[0], event_pts_x[2], alpha=0.2, color='k', linewidth=0)
        ax.axvspan(event_pts_x[2], _len, alpha=0.1, color='k', linewidth=0)
        if ax_idx == 0:
            ax_pos_y = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.26
        else:
            ax_pos_y = ax.get_ylim()[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.31
        ax.text(intersects_x[0] / 2, ax_pos_y, 'upstream', horizontalalignment='center', rotation=0,
                fontsize=FSIZE_TINY)
        ax.text(intersects_x[0] + (event_pts_x[2] - intersects_x[0]) / 2, ax_pos_y, 'downstream',
                horizontalalignment='center', rotation=0, fontsize=FSIZE_TINY)
        ax.text(event_pts_x[2] + (_len - event_pts_x[2]) / 2, ax_pos_y, 'normalization', horizontalalignment='center',
                rotation=0, fontsize=FSIZE_TINY)
    for ax in axs.flatten():
        ax.set_xticks([25 * i for i in range(int(_len / 25) + 1)])
    for ax in axs.flatten():
        # ax.set_xticks([0])
        ax.set_xticklabels(['0'])
        ax.set_yticks([0])
        ax.set_yticklabels(['0'])
    # for ax in axs[:3, :].flatten():
    #     ax.locator_params(tight=True, nbins=4, axis='y')

    plt.tight_layout()

    for idx, ax in enumerate(axs.flatten()):
        pos_old = ax.get_position()
        x_new = 0.13
        y_new = pos_old.y0 - 0.03 * idx
        width_new = 1 - x_new
        height_new = pos_old.height * 1.05
        ax.set_position([x_new, y_new, width_new, height_new])
        fig.text(0, ax.get_position().y0 + ax.get_position().height, chr(idx + 97), horizontalalignment='left',
                 verticalalignment='center', fontweight='bold')

    title_y_top = axs[0].get_position().y0 + axs[0].get_position().height / 2
    title_y_bottom = axs[1].get_position().y0 + axs[1].get_position().height / 2
    fig.text(0, title_y_top, 'Consumption Price', horizontalalignment='left', verticalalignment='center', rotation=90)
    fig.text(0, title_y_top, '\n(%baseline)', horizontalalignment='left', verticalalignment='center', rotation=90)
    fig.text(0, title_y_bottom, 'Consumption', horizontalalignment='left', verticalalignment='center', rotation=90)
    fig.text(0, title_y_bottom, '\n(%baseline)', horizontalalignment='left', verticalalignment='center', rotation=90)

    center_x = axs[0].get_position().x0 + axs[0].get_position().width / 2
    fig.text(center_x, 0, 'days', horizontalalignment='center', verticalalignment='bottom')

    if _outfile is not None:
        fig.savefig(_outfile, dpi=300)

    if _show:
        plt.show()


def make_consumption_figure_for_paper_with_schematic(_data, _clip, _t0=0, _outfile=None, _show=True):
    scale_factor = 1.0
    fig_width = MAX_FIG_WIDTH_NARROW * scale_factor
    fig_height = 6.5

    us_states_linewidth = 0.5
    regions_linewidth = 1

    prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prop_cycle_colors = prop_cycle_colors[:3] + prop_cycle_colors[4:]
    colors = {}
    for idx, r in enumerate(['US.NJ', 'US.NY', 'CAN', 'MEX', 'CHN', 'DEU', 'EUR']):
        colors[r] = prop_cycle_colors[idx]

    fig, axs = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(fig_width, fig_height))

    for ax in axs.flatten():
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    us_state_list = [r for r in _data.get_regions() if r[:3] == 'US.' and r not in ['US.NJ', 'US.NY']]
    greys_map = plt.cm.get_cmap('Greys')
    for col_idx, var in enumerate(['consumption_price', 'consumption']):
        greys = [greys_map(0.2 + (0.7 * i) / len(us_state_list)) for i in range(len(us_state_list))]
        for state in us_state_list:
            color = greys[-1]
            baseline = _data.get_regions(state).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()[0]
            time_series = _data.get_regions(state).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()
            time_series = (time_series / baseline - 1) * 100
            axs[0, col_idx].plot(time_series[_t0:], color=color, linewidth=us_states_linewidth)
            greys = greys[:-1]

    # observed data:
    region_lists = [['US.NJ', 'US.NY'], ['MEX', 'CAN'], ['EUR', 'DEU', 'CHN']]
    for col_idx, var in enumerate(['consumption_price', 'consumption']):
        for row_idx, region_list in enumerate(region_lists):
            axs[row_idx, col_idx].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            for region in region_list:
                color = colors[region]
                label = region
                if region in ['US.NJ', 'US.NY']:
                    label = region[3:]
                baseline = _data.get_regions(region).get_vars(var).get_sectors('FCON').clip(_clip).get_data().flatten()[
                    0]
                time_series = _data.get_regions(region).get_vars(var).get_sectors('FCON').clip(
                    _clip).get_data().flatten()
                time_series = (time_series / baseline - 1) * 100
                axs[row_idx, col_idx].plot(time_series[_t0:], color=color, linewidth=regions_linewidth, label=label)
            loc = 'lower right'
            if row_idx == 2:
                loc = 'upper right'
            axs[row_idx, 1].legend(frameon=False, loc=loc, labelspacing=0.25, handlelength=1.5, borderpad=0.2)

    # schematic plots:
    pts_s = [0, 10, 50, 60, 100, 113, 125, 130]
    pts_y = [0, -0.05, 0.2, 0.2, 0.03, 0.013, 0.01, 0.01]
    f = interp1d(pts_s, pts_y, fill_value='extrapolate', kind='cubic')
    x = np.arange(_clip)
    y = f(np.arange(_clip))
    axs[-1, 0].plot(x, y, linewidth=regions_linewidth, color='k')
    axs[-1, 1].plot(np.arange(_clip), y * -0.5, linewidth=regions_linewidth, color='k')
    event_pts_y = [0, min(y), max(y)]
    event_pts_x = [list(y).index(pt) for pt in event_pts_y]
    intersects_x = [i for i in range(1, len(y) - 1) if y[i] * y[i + 1] < 0]
    # axs[-1, 0].scatter(event_pts_x, y[event_pts_x], s=3, marker='o', c='w', edgecolors='k', linewidths=1)
    # axs[-1, 0].arrow(0, 0.1, 0, -0.1, shape='full', length_includes_head=True, color='grey', width=1, head_width=5)
    # axs[-1, 0].plot(0, 0.01, 'v', markersize=3, c='grey')
    # axs[-1, 0].plot([0, 0.01], [0, 0.03], c='grey')
    # axs[-1, 0].text(0, 0.04, 'disaster', rotation=90, fontsize=tiny_fsize, horizontalalignment='center', verticalalignment='bottom')
    for ax_idx, ax in enumerate(axs[-1, :]):
        ax.axvspan(0, intersects_x[0], alpha=0.1, color='k', linewidth=0)
        ax.axvspan(intersects_x[0], event_pts_x[2], alpha=0.2, color='k', linewidth=0)
        ax.axvspan(event_pts_x[2], _clip, alpha=0.1, color='k', linewidth=0)
        ax_center = ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5  # (0.6 if ax_idx == 0 else 0.4)
        texts = []
        texts.append(ax.text(intersects_x[0] / 2, ax_center, 'upstream', horizontalalignment='center',
                             verticalalignment='center', rotation=90, fontsize=FSIZE_TINY))
        texts.append(ax.text(intersects_x[0] + (event_pts_x[2] - intersects_x[0]) / 2, ax_center, 'downstream',
                             horizontalalignment='center', verticalalignment='center', rotation=90,
                             fontsize=FSIZE_TINY))
        texts.append(ax.text(event_pts_x[2] + (_clip - event_pts_x[2]) / 2, ax_center, 'normalization',
                             horizontalalignment='center', verticalalignment='center', rotation=90,
                             fontsize=FSIZE_TINY))
        for t in texts:
            t.set_bbox(dict(facecolor='white', alpha=0.75, linewidth=0))
    for ax in axs.flatten():
        ax.set_xticks([25 * i for i in range(int(_clip / 25) + 1)])
    for ax in axs[:2, :].flatten():
        ax.set_xticklabels([])
    for ax in axs[3, :].flatten():
        # ax.set_xticks([0])
        ax.set_xticklabels(['0'])
        ax.set_yticks([0])
        ax.set_yticklabels([0.00])
    # for ax in axs[:3, :].flatten():
    #     ax.locator_params(tight=True, nbins=4, axis='y')

    axs[2, 0].set_ylim(axs[2, 0].get_ylim()[0], axs[2, 0].get_ylim()[1] * 1.3)
    axs[1, 0].set_ylim(axs[1, 0].get_ylim()[0], axs[1, 0].get_ylim()[1] * 1.15)

    plt.tight_layout()

    for idx, ax in enumerate(axs.flatten()):
        pos_old = ax.get_position()
        x_new = pos_old.x0
        y_new = pos_old.y0
        width_new = pos_old.width * 1.1
        height_new = pos_old.height * 1.02
        if idx in [0, 1]:
            y_new = pos_old.y0 - 0.035
        if idx in [2, 3]:
            y_new = pos_old.y0 - 0.01
        if idx in [4, 5]:
            y_new = pos_old.y0 + 0.015
        # if idx in [6, 7]:
        #     y_new = pos_old.y0 - 0.05
        if idx % 2 == 0:
            x_new = 0.11
        else:
            x_new = 1 - width_new
        ax.set_position([x_new, y_new, width_new, height_new])
        fig.text(x_new - 0.1, ax.get_position().y0 + ax.get_position().height, chr(idx + 97),
                 horizontalalignment='left',
                 verticalalignment='center', fontweight='bold')

    center_x_left = axs[0, 0].get_position().x0 + axs[0, 0].get_position().width / 2
    center_x_right = axs[0, 1].get_position().x0 + axs[0, 1].get_position().width / 2

    # title_y = axs[0, 0].get_position().y0 + axs[0, 0].get_position().height * 1.05
    fig.text(center_x_left, 1, 'Consumption price\n(% baseline)', horizontalalignment='center', verticalalignment='top',
             fontweight='bold')
    fig.text(center_x_right, 1, 'Consumption\n(% baseline)', horizontalalignment='center', verticalalignment='top', fontweight='bold')

    xlabel_y = axs[-1, 0].get_position().y0 - 0.04
    fig.text(center_x_left, xlabel_y, 'days', horizontalalignment='center')
    fig.text(center_x_right, xlabel_y, 'days', horizontalalignment='center')

    if _outfile is not None:
        fig.savefig(_outfile, dpi=300)

    if _show:
        plt.show()


def make_time_series_figure_for_paper(_data: AggrData, _clip, _t0=0, _outfile=None, _show=True):
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

    prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    prop_cycle_colors = prop_cycle_colors[:3] + prop_cycle_colors[4:]
    colors = {}
    for idx, r in enumerate(regions_to_plot):
        colors[r] = prop_cycle_colors[idx]

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

    for ax_idx, var in enumerate(vars_to_plot):
        for region in regions_to_plot:
            ts = _data.get_vars(var).get_regions(region).get_sectors('PRIVSECTORS').get_data().flatten()
            if var in ['effective_production_capacity', 'production_price']:
                ts = (ts / ts[0]) - 1
            axs[ax_idx].plot(ts[_t0:] * 100, color=colors[region], linewidth=1.5, label=regions_to_plot[region])
    axs[0].legend(frameon=False, loc='upper left', labelspacing=0.25, bbox_to_anchor=(1.01, 1.01))

    plt.tight_layout()

    axs[1].set_ylim((axs[1].get_ylim()[0], 225))
    axs[2].set_ylim((-7, axs[2].get_ylim()[1]))

    for ax_idx, label in enumerate(list(vars_to_plot.values())):
        axis_pos = axs[ax_idx].get_position()
        axs[ax_idx].set_position([axis_pos.x0 + 0.04, axis_pos.y0, axis_pos.width - 0.01, axis_pos.height])
        y_pos = axs[ax_idx].get_position().y0 + axs[ax_idx].get_position().height / 2
        fig.text(0, y_pos, label, verticalalignment='center', fontsize=FSIZE_SMALL, rotation=90)
        fig.text(0, y_pos, "\n(% baseline)", verticalalignment='center', fontsize=FSIZE_SMALL, rotation=90)
        fig.text(0, axs[ax_idx].get_position().y0 + axs[ax_idx].get_position().height, chr(ax_idx + 97),
                 horizontalalignment='left', verticalalignment='top', fontweight='bold')

    if _outfile is not None:
        fig.savefig(_outfile, dpi=300)
    if _show:
        plt.show()


def make_schematic_time_series_figure_for_paper(_outfile=None, _show=True):
    _t0 = 4
    _data = pickle.load(open("../data/acclimate_output/sandy_data_first_draft__data.pk", "rb"))
    scale_factor = 1.0
    fig_width = MAX_FIG_WIDTH_NARROW * scale_factor
    fig_height = 6

    _len = 180

    _data = copy.deepcopy(_data.get_regions('US.NJ'))
    _data.drop_var('production_price', _inplace=True)
    _data.calc_prices(_inplace=True)
    _data.calc_eff_forcing(_inplace=True)
    _data.calc_eff_prod_capacity(_inplace=True)
    _data.calc_demand_exceedence(_inplace=True)

    variables = ['production_price', 'demand_exceedence', 'effective_production_capacity']
    variable_names = ['Production price', 'Demand exceedance', 'Capacity utilization']

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
            'xy': [(141 - _t0, -0.17967579433559078), (142 - _t0, 0.07921862316064199)],
            'labels': ['demand drop', 'demand overshoot'],
            'xytext': [(110, -0.45), (150, 0.4)],
        },
        'effective_production_capacity': {
            'xy': [(141 - _t0, -3.179425279153936e-05), (142 - _t0, -0.075)],
            'labels': ['production regime\nchange', 'production drop'],
            'xytext': [(150, 0.025), (150, -0.1)],
        },
    }

    y_lims = {
        'production_price': (-0.06, 0.07),
        'demand_exceedence': (-0.5, 2.2),
        'effective_production_capacity': (-0.12, 0.05),
    }

    fig, axs = plt.subplots(3, 1, sharex=True, sharey=False, figsize=(fig_width, fig_height))

    for ax in axs.flatten():
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)

    for ax_idx, ax in enumerate(axs):
        var = variables[ax_idx]
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
        ax.plot(list(ts[_t0:]) + [0] * (_len - _clip), color='k', linewidth=1)
        ax.set_ylim(y_lims[var])
        # shading_areas[var]['y'] = [ts[x] for x in shading_areas[var]['x']]
        # annotation_points[var]['y'] = [ts[x] for x in annotation_points[var]['x']]
        for area_idx, area_label in enumerate(shading_areas[var]['labels']):
            ax.axvspan(shading_areas[var]['x'][area_idx], shading_areas[var]['x'][area_idx + 1],
                       alpha=0.1 + 0.1 * area_idx % 2, color='k', linewidth=0)
            ax.text(shading_areas[var]['x'][area_idx] + (
                    shading_areas[var]['x'][area_idx + 1] - shading_areas[var]['x'][area_idx]) / 2,
                    ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * shading_areas[var]['y_pos'][area_idx],
                    area_label, rotation=shading_areas[var]['rotation'][area_idx],
                    horizontalalignment='center', verticalalignment='center', fontsize=FSIZE_TINY)
        for annotation_idx, annotation_label in enumerate(annotation_points[var]['labels']):
            xy = annotation_points[var]['xy'][annotation_idx]
            xytext = annotation_points[var]['xytext'][annotation_idx]
            ax.annotate(annotation_label, xy, xytext=xytext, arrowprops={'arrowstyle': '->'},
                        horizontalalignment='center', fontsize=FSIZE_TINY)
    for ax in axs.flatten():
        ax.set_xticks([25 * i for i in range(int(_len / 25) + 1)])
    for ax in axs.flatten():
        # ax.set_xticks([0])
        ax.set_xticklabels([])
        ax.set_yticks([0])
        ax.set_yticklabels(['0'])
    plt.tight_layout()

    for idx, ax in enumerate(axs.flatten()):
        pos_old = ax.get_position()
        x_new = 0.1
        y_new = pos_old.y0 + 0.05 * (idx + 1)
        width_new = 1 - x_new
        height_new = pos_old.height * 0.85
        ax.set_position([x_new, y_new, width_new, height_new])
        fig.text(0, ax.get_position().y0 + ax.get_position().height, chr(idx + 97), horizontalalignment='left',
                 verticalalignment='center', fontweight='bold')

    for ax_idx, ax in enumerate(axs):
        title_y = ax.get_position().y0 + axs[0].get_position().height / 2
        fig.text(0, title_y, variable_names[ax_idx], horizontalalignment='left', verticalalignment='center',
                 rotation=90)

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
    time_ax.set_xticklabels(['0'])
    time_ax.set_yticks([])
    time_ax.set_yticklabels(['0'])
    time_ax.axvline(x=time_ax.get_xlim()[0], ymin=0, ymax=0.5, color='k', linewidth=plt.rcParams['axes.linewidth'])
    fig.text(0, time_ax_pos[1] + time_ax_pos[3] * 0.25, 'Timeline', horizontalalignment='left',
             verticalalignment='center', rotation=90)
    fig.text(0, time_ax_pos[1] + time_ax_pos[3], chr(3 + 97), horizontalalignment='left',
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
    # ha='center', va='bottom', arrowprops=dict(arrowstyle='-[, widthB={}, lengthB={}'.format(141 - _t0, 1.5), lw=1.0))

    center_x = axs[-1].get_position().x0 + axs[0].get_position().width / 2
    fig.text(center_x, 0, 'time', horizontalalignment='center', verticalalignment='bottom')

    if _outfile is not None:
        fig.savefig(_outfile, dpi=300)

    if _show:
        plt.show()


def make_consumption_deviation_map(_data, _clip, _t0, _outfile=None, _excluded_countries=None, _only_usa=False,
                                   _numbering=None, _csv_outfile=None):
    scale_factor = 0.75

    prop_cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if _excluded_countries is None:
        _excluded_countries = []
    _data = _data.get_sectors('FCON').get_vars('consumption')
    consumption = _data.get_data()[..., _t0:_clip]
    consumption_losses = consumption.sum(axis=-1).flatten() - consumption[..., 0].flatten() * (_clip - _t0)
    consumption_deviation = consumption_losses / (consumption[..., 0].flatten() * (_clip - _t0))
    consumption_deviation = consumption_deviation * 100
    regions = _data.get_regions()

    df = pd.DataFrame()
    df['region'] = regions
    df['consumption_deviation'] = consumption_deviation
    df.set_index('region', drop=True)
    print(df.loc[['USA', 'MEX', 'CAN', 'EUR', 'DEU', 'CHN', 'WORLD', 'ROW']])

    # df_reduced = df.loc[list(set(WORLD_REGIONS['WORLD']) - {'USA', 'CHN'} - set(_excluded_countries))]
    df_reduced = df.loc[list(set(WORLD_REGIONS['WORLD']) - {'USA', 'CHN'})]
    df_reduced.loc[_excluded_countries, 'consumption_deviation'] = np.nan
    print(df_reduced.index)

    if _only_usa:
        df_reduced = df_reduced[df_reduced['region'].apply(lambda x: True if x[:3] in ['USA', 'US.'] else False)]

    if df_reduced['consumption_deviation'].max() > 0:
        # cm = create_colormap("custom", [prop_cycle_colors[3], "white", prop_cycle_colors[0]], xs=[0, -df_reduced['consumption_deviation'].min() / (df_reduced['consumption_deviation'].max() - df_reduced['consumption_deviation'].min()), 1])
        cm = create_colormap("custom", ['#f64748', "white", 'royalblue'], xs=[0, -df_reduced[
            'consumption_deviation'].min() / (df_reduced['consumption_deviation'].max() - df_reduced[
            'consumption_deviation'].min()), 1])
    else:
        # cm = create_colormap("custom", [mpl.colors.hsv_to_rgb((0, 1, 1)), mpl.colors.hsv_to_rgb((0, 0.2, 1))], xs=[0, 1])
        cm = create_colormap("custom", ['#f64748', '#f6d5d5'], xs=[0, 1])

    fig = plt.figure(figsize=(MAX_FIG_WIDTH_WIDE * scale_factor, MAX_FIG_WIDTH_WIDE * scale_factor / 2.2))
    gs = plt.GridSpec(1, 2, width_ratios=[1, 0.03])
    cax = gs[0, 1]
    ax = gs[0, 0]
    if _only_usa:
        # patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/map_robinson_0.1simplified_only_US.pkl.gz"
        # patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/usa.pkl.gz"
        patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/usa_new.pkl.gz"
        # patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/usa_new_ak_hi.pkl.gz"
        # lims = (25, 50, -115, -49.5)
        lims = (-13, 15, -34, 28)
        # lims=None
    else:
        patchespickle_file = "/home/robin/repos/hurricanes_hindcasting_remake/global_map/map_robinson_0.1simplified.pkl.gz"
        lims = None
    make_map(patchespickle_file=patchespickle_file,
             regions=df_reduced['region'],
             data=df_reduced['consumption_deviation'],
             y_ticks=None,
             y_label='Consumption deviation \n(% baseline)',
             numbering=_numbering,
             numbering_fontsize=FSIZE_SMALL,
             extend_c="both",
             ax=fig.add_subplot(ax),
             cax=fig.add_subplot(cax),
             cm=cm,
             y_label_fontsize=FSIZE_SMALL,
             y_ticks_fontsize=FSIZE_SMALL,
             ignore_regions=None,
             lims=lims,
             only_usa=_only_usa,
             )
    fig.tight_layout()
    if isinstance(_outfile, str):
        fig.savefig(_outfile, dpi=300)
    if isinstance(_csv_outfile, str):
        df_reduced.to_csv(_csv_outfile)
    plt.show()


def make_import_dependency_figure_for_paper(_data: AggrData, _clip=None, _t0=None, _outfile=None):
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
        fig.savefig(_outfile, dpi=300)
    plt.show()


if __name__ == '__main__':
    data = pickle.load(open("../data/acclimate_output/sandy_data_first_draft__data.pk", "rb"))
    # make_consumption_figure_for_paper_with_schematic(data, 124, 4, _outfile="../figures/consumption_plots.pdf")
    # make_consumption_deviation_map(data, _clip=104, _t0=4, _numbering='a', _excluded_countries=['BLR', 'SOM', 'ETH'],
    #                                _outfile="../figures/choropleth_consumption_deviation_WORLD.pdf")
    # make_consumption_deviation_map(data, _clip=104, _t0=4, _only_usa=True, _numbering='b',
    #                                _outfile="../figures/choropleth_consumption_deviation_USA.pdf")
    # make_import_dependency_figure_for_paper(data, _clip=104, _t0=4,
    #                                         _outfile='../figures/consumption_deviation__trade_dependencies.pdf')
    # make_schematic_time_series_figure_for_paper(_outfile="../figures/time_series_schematic.pdf", _show=True)
    # make_time_series_figure_for_paper(data, _clip=104, _t0=4, _outfile="../figures/time_series_NJ_NY_US-49.pdf")


