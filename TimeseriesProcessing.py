import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
# from config import TIMESTEP_STRING
import argparse
import sys

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
# data_path = 'data/CSV'

data_dict = joblib.load('data/data_dict')

# data_sources_list = ['gamelog', 'hrm', 'envibox', 'datalog'] # , 'eyetracker', 'key', 'mkey', 'mxy', 'gyro']  # List sources for analysis here
data_dict_resampled = {}

movements_columns = ['gaze_movement', 'mouse_movement', 'mouse_scroll']  # Using mean for the resampling is not correct
# because the result depends on the number of samples

# print(sys.argv)
# del sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument('--TIMESTEP_STRING', default='10s', type=str)
args = parser.parse_args()
TIMESTEP_STRING = args.TIMESTEP_STRING

for player_id, player_data_dict in data_dict.items():
    if 'gamelog' not in player_data_dict:
        continue

    # player_id = '9'  ### DEBUG
    # player_data_dict = data_dict[player_id]  ### DEBUG

    player_data_dict_resampled = {}

    df_resampled4player = pd.DataFrame()

    for data_source in player_data_dict.keys():
        if data_source == 'gamelog':
            # player_data_dict_resampled['gamelog'] = player_data_dict['gamelog']
            continue

        # data_source = 'envibox'  ### DEBUG

        if 'time' not in player_data_dict[data_source].columns:
            print('Cant see time in ', data_source)
        # else:
        #     print('See time in ', data_source)

        df = player_data_dict[data_source]
        # df = df.copy()  # DEBUG

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        if data_source == 'envibox':
            # 'als' column should be dropped here
            df.drop(columns=['mic', 'als'], inplace=True)

        if data_source == 'eyetracker':
            # df['gaze_x'] /= SCREEN_SIZE_X
            # df['gaze_y'] /= SCREEN_SIZE_Y   # Normalization to different constants is strange

            df['x_diff'] = np.append(0, np.diff(df['gaze_x'].values))
            df['y_diff'] = np.append(0, np.diff(df['gaze_y'].values))
            df['gaze_movement'] = (df['x_diff'] ** 2 + df['y_diff'] ** 2) ** 0.5

            df.drop(columns=['gaze_x', 'gaze_y', 'x_diff', 'y_diff'], inplace=True)

            # plt.plot(df['x_diff'][:10000])
            # plt.plot(df['diff'][:10000])

            # plt.close()
            # plt.hist(df['diff'], bins=1000, range=(0, 100))
            # df['diff'].quantile(0.)

        if data_source == 'mxy':
            # plt.close()
            # plt.hist(df['mouse_dy'], range=(-20, 20))
            # df['mouse_dy']
            # df['mouse_dy'].min()
            # df['mouse_dx'].max()
            df['mouse_movement'] = (df['mouse_dx'] ** 2 + df['mouse_dy'] ** 2) ** 0.5
            df['mouse_scroll'] = df['mouse_scroll'].abs()
            df.drop(columns=['mouse_dx', 'mouse_dy'], inplace=True)

        if data_source == 'datalog':
            df.drop(columns=['hrm2'], inplace=True)  # I don't know how to process it yet

            serial = df['resistance'] * 1024 / 3.3
            serial_max = 768  # I'm just guessing
            # print(serial.max())  # Really need to be checked
            df['resistance'] = (1024 + 2 * serial) * 10000 / (serial_max - serial)


        if data_source == 'schairlog':
            df.drop(columns=['mag_x', 'mag_y', 'mag_z'], inplace=True)


        # df = df.drop_duplicates()
        unique_values, unique_indexes = np.unique(df.index, return_index=True)
        df = df.iloc[unique_indexes, :]
        # plt.plot(df.index)

        if data_source in movements_columns:
            df_resampled = df.resample(TIMESTEP_STRING, 'sum')
        else:
            df_resampled = df.resample(TIMESTEP_STRING, 'mean')

        # df_resampled = df_resampled.join(df_resampled_movements)


        # .median()  # It's better I think, because there are no NaNs
        # df_resampled = df.resample('100ms').median()  # .mean()
        # print(df_resampled.isnull().mean())

        # player_data_dict_resampled[data_source] = df_resampled
        # df_resampled_list.append(df_resampled)
        df_resampled4player = df_resampled4player.join(df_resampled, how='outer')


        # player_data_dict_resampled[data_source] = df.resample('100ms', fill_method='nearest')#.mean()
        # df.fillna(WHAT)

    df_resampled4player.interpolate(method='linear', inplace=True)
    # player_data_dict_resampled['data'] = df_resampled4player
    # data_dict_resampled[player_id] = player_data_dict_resampled
    data_dict_resampled[player_id] = df_resampled4player




joblib.dump(data_dict_resampled, 'data/data_dict_resampled')















# class TimeseriesProcesser:
#
#     def __init__(self, df, sample_rate):
#         self.df = df
#         self.sample_rate = sample_rate
#
#     def resample(self):
#












