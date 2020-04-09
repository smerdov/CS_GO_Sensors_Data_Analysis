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

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
pic_folder = 'pic/'

parser = argparse.ArgumentParser()
parser.add_argument('--TIMESTEP', default=10, type=float)
parser.add_argument('--plot', default=0, type=int)
if __debug__:
    print('SUPER WARNING!!! YOU ARE INTO DEBUG MODE', file=sys.stderr)
    args = parser.parse_args(['--TIMESTEP=10', '--plot=1'])
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
window_sizes_list = [60, 120, 180, 300, 600]

# player_id = '0'
for player_id in data_dict_resampled_merged:
    df_merged = data_dict_resampled_merged[player_id]

    mask_negative = df_merged.index < pd.to_timedelta(0)  # I just don't fucking care about that
    if mask_negative.sum():
        print('EMERGENCY!')
        break

    df_merged = df_merged.loc[~mask_negative]


    for window_size in window_sizes_list:
        # window_size = 180
        target_colname = f'kills_proportion_{window_size}'

        window_steps = int(window_size // TIMESTEP)
        # df_stats = df_merged[['kill', 'death']].rolling(f'{window_size}s', min_periods=window_steps).sum()
        df_stats = df_merged[['kill', 'death']].rolling(f'{window_size}s', min_periods=window_steps // 2).sum()
        df_stats[target_colname] = df_stats['kill'] / (df_stats['death'] + df_stats['kill'])

        print(target_colname)
        print(df_stats[target_colname])

        if plot:
            plt.plot(df_stats[target_colname], label=window_size)

        df_merged[target_colname + '_4future'] = None
        df_merged[target_colname + '_4past'] = None
        df_merged[target_colname + '_4future'].iloc[:-window_steps] = df_stats[target_colname].iloc[window_steps:].values
        df_merged[target_colname + '_4past'].iloc[window_steps:] = df_stats[target_colname].iloc[window_steps:].values  # original

    if plot:
        plt.legend()
        plt.tight_layout()
        plt.savefig(pic_folder + f'player_{player_id}_performance.png')
        plt.close()

    # df_stats['kills_proportion'].max()

    # square_plot(df_merged, df_merged.columns)

    df4train = df_merged.drop(columns=['kill', 'death', 'shootout'])# .iloc[:-window_steps]

    # plt.plot(df4train['kills_proportion'])

    data_dict_resampled_merged_with_target[player_id] = df4train


joblib.dump(data_dict_resampled_merged_with_target, 'data/data_dict_resampled_merged_with_target')




# square_plot(df, columns2plot=df.columns[1:], suffix=arduino_name)











