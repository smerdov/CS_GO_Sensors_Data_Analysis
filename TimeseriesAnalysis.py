import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import string2json
# from config import TIMESTEP
import itertools
import argparse
import sys
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
pic_folder = 'pic/'

parser = argparse.ArgumentParser()
parser.add_argument('--TIMESTEP', default=20, type=float)
parser.add_argument('--plot', default=0, type=int)
if __debug__:
    print('SUPER WARNING!!! YOU ARE INTO DEBUG MODE', file=sys.stderr)
    args = parser.parse_args(['--TIMESTEP=20', '--plot=1'])
else:
    args = parser.parse_args()

TIMESTEP = args.TIMESTEP
plot = args.plot

data_dict_resampled_merged = joblib.load('data/data_dict_resampled_merged')

data_dict_resampled_merged_with_target = {}

# player_id = list(data_dict_resampled_merged.keys())[0]  # DEBUG

def square_plot(df, columns2plot, timecol='Timestamp', suffix='last'):
    n_plots = len(columns2plot)
    square_size = int(np.ceil(n_plots ** 0.5))
    # time_data = df[timecol]
    time_data = df.index
    fig, ax = plt.subplots(square_size, square_size, sharex=False, sharey=False, figsize=(20, 20))
    fig.suptitle(suffix)

    for n_plot, (n_row, n_col) in enumerate(itertools.product(range(square_size), range(square_size))):
        if n_plot >= n_plots:
            continue

        colname = columns2plot[n_plot]
        ax[n_row, n_col].plot(time_data.values, df[colname].values)
        ax[n_row, n_col].set_title(colname)

    fig.tight_layout()
    fig.savefig(f'{pic_folder}square_plot_{suffix}.png')

# player_id = '10'  # DEBUG
# window_sizes_list = [60, 120, 180, 300, 600]
# window_sizes_list = [30, 60, 120, 180, 240, 300]
# window_sizes_list = [60, 120, 180, 240, 300]
window_sizes_list = [60, 180, 300]


# player_id = '19'
for player_id in data_dict_resampled_merged:
    df_merged = data_dict_resampled_merged[player_id]

    mask_negative = df_merged.index < pd.to_timedelta(0)  # I just don't fucking care about that
    print(mask_negative.sum())
    if mask_negative.sum():
        print('EMERGENCY!')
        break

    df_merged = df_merged.loc[~mask_negative]

    fontsize = 20
    lw = 5
    alpha = 0.8
    markersize = 15
    n_skipped_points = 10
    colors_dict = {
        60: 'olivedrab',
        180: 'dodgerblue', # 'lightseagreen', # 'turquoise',# ,  # ,
        300: 'firebrick',
    }

    plt.close()
    # fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(9, 5), squeeze=False)
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(14, 8), squeeze=False, sharex=True,
                           gridspec_kw={'height_ratios': [1, 0.2, 0.2, 0.2]})


    # window_size = 300
    for n_window_size, window_size in enumerate(window_sizes_list):
        target_colname = f'kills_proportion_{window_size}'

        window_steps = int(window_size // TIMESTEP)
        # df_stats = df_merged[['kill', 'death']].rolling(f'{window_size}s', min_periods=window_steps).sum()
        min_steps = window_steps // 2  # TODO: check. 10 apr
        df_stats = df_merged[['kill', 'death']].rolling(f'{window_size}s', min_periods=min_steps).sum()
        df_stats[target_colname] = df_stats['kill'] / (df_stats['death'] + df_stats['kill'])

        # print(target_colname)
        # print(df_stats[target_colname])

        if plot:
            values4plot = df_stats[target_colname].copy()
            values4plot = values4plot.values[n_skipped_points:]
            values4plot_notnull = ~np.isnan(values4plot)
            values4plot[~values4plot_notnull] = 0
            n_samples_cum = np.cumsum(values4plot_notnull)

            indexes = np.arange(len(values4plot)) * TIMESTEP
            # label = f'Predict Lenght = {window_size}s'
            # label = f'{window_size}s average'
            label = f'$\\tau = {window_size}$ s'
            color = colors_dict[window_size]
            ax[0, 0].plot(indexes[values4plot_notnull], values4plot[values4plot_notnull], label=label, lw=lw, c=color, alpha=alpha)

            # target_tmp = values4plot.values
            # floating_mean = np.cumsum(values4plot) / np.arange(1, len(values4plot) + 1)
            floating_mean = np.cumsum(values4plot) / n_samples_cum
            target = values4plot - floating_mean
            # target = values4plot[values4plot_notnull] - floating_mean[values4plot_notnull]

            binary_target = (target > 0) * 1
            # binary_target = values4plot.values
            # ax[3, 0].plot(binary_target, label='Binary Target', color='peru')
            breakpoints = list(np.nonzero(np.diff(binary_target) != 0)[0])
            breakpoints = sorted(breakpoints)
            # print(breakpoints)
            if 0 not in breakpoints:
                breakpoints = [0] + breakpoints
                zero_fake = True
            else:
                zero_fake = False
            if len(binary_target) - 1 not in breakpoints:
                breakpoints = breakpoints + [len(binary_target) - 1]
                last_fake = True
            else:
                last_fake = False

            for n_breakpoint in range(len(breakpoints) - 1):
                if (n_breakpoint == 0) and zero_fake:
                    breakpoint_start = breakpoints[n_breakpoint]
                else:
                    breakpoint_start = breakpoints[n_breakpoint] + 1
                breakpoint_end = breakpoints[n_breakpoint + 1]
                x_points = list(range(breakpoint_start, breakpoint_end + 1))
                ax[n_window_size + 1, 0].plot(np.array(x_points) * TIMESTEP, binary_target[breakpoint_start:breakpoint_end + 1],
                              label='Binary Target', color=color, lw=lw*0.66)
            ax[n_window_size + 1, 0].scatter(np.array(list(range(len(target)))) * TIMESTEP, binary_target, s=markersize, color=color)

        df_merged[target_colname + '_4future'] = None
        df_merged[target_colname + '_4past'] = None
        df_merged[target_colname + '_4future'].iloc[:-window_steps+1] = df_stats[target_colname].iloc[window_steps-1:].values
        # df_merged[target_colname + '_4past'].iloc[window_steps:] = df_stats[target_colname].iloc[window_steps:].values  # original
        df_merged[target_colname + '_4past'].iloc[window_steps - min_steps-1:] = df_stats[target_colname].iloc[window_steps-min_steps-1:].values  # original

    if plot:
        ax[0, 0].set_xlim(0 - TIMESTEP / 2, indexes.max() + TIMESTEP / 2)
        # ax.set_ylabel('Kills Ratio', fontsize=fontsize+2)
        # ax.set_ylabel('Player Performance $p_t(t)$', fontsize=fontsize+2)


        for n_row in range(4):
            ax[n_row, 0].xaxis.set_major_locator(MultipleLocator(500))
            ax[n_row, 0].xaxis.set_minor_locator(MultipleLocator(100))
            ax[n_row, 0].tick_params(axis='both', which='major', labelsize=fontsize - 2, size=fontsize*0.52)
            ax[n_row, 0].tick_params(axis='both', which='minor', size=fontsize*0.28)
            ax[n_row, 0].yaxis.set_label_coords(-0.046, 0.5)

            if n_row == 0:
                ax[n_row, 0].yaxis.set_major_locator(MultipleLocator(0.2))
                ax[n_row, 0].yaxis.set_minor_locator(MultipleLocator(0.1))
                ax[n_row, 0].legend(fontsize=fontsize-2, loc='lower right')
                ax[n_row, 0].set_title(f'Player Performance', fontsize=fontsize + 2)
                ax[n_row, 0].set_ylabel('$p_\\tau(t)$', fontsize=fontsize + 2)
            else:
                window_size4plot = window_sizes_list[n_row-1]
                # ax[n_row, 0].set_title(f'Binary Target for $\\tau={window_size4plot}$', fontsize=fontsize + 2)
                ax[n_row, 0].set_title(f'Binary Target, $\\tau={window_size4plot}$', fontsize=fontsize + 2)
                ax[n_row, 0].yaxis.set_major_locator(MultipleLocator(1))
                ax[n_row, 0].set_ylabel('$y_\\tau(t)$', fontsize=fontsize + 2)

            if n_row == 3:
                ax[n_row, 0].set_xlabel('Time $t$, s', fontsize=fontsize+2)

        # plt.legend(fontsize=fontsize)
        # plt.tight_layout(rect=[-0.01, -0.032, 1.012, 1.021])
        plt.tight_layout(rect=[-0.004, -0.015, 1.012, 1.015])

        # plt.savefig(pic_folder + f'player_{player_id}_performance.png')
        plt.savefig(pic_folder + f'player_{player_id}_performance.pdf')
        plt.close()

    # df_stats['kills_proportion'].max()

    # square_plot(df_merged, df_merged.columns)

    df4train = df_merged.drop(columns=['kill', 'death', 'shootout'])# .iloc[:-window_steps]

    # plt.plot(df4train['kills_proportion'])

    data_dict_resampled_merged_with_target[player_id] = df4train


joblib.dump(data_dict_resampled_merged_with_target, 'data/data_dict_resampled_merged_with_target')




# square_plot(df, columns2plot=df.columns[1:], suffix=arduino_name)











