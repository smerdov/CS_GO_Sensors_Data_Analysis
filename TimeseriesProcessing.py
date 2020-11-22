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



parser = argparse.ArgumentParser()
parser.add_argument('--TIMESTEP_STRING', default='10s', type=str)
if __debug__:
    print('SUPER WARNING!!! YOU ARE INTO DEBUG MODE', file=sys.stderr)
    args = parser.parse_args(['--TIMESTEP_STRING=10s'])
else:
    args = parser.parse_args()

TIMESTEP_STRING = args.TIMESTEP_STRING

def clip_by_percentile(x, percentile=5):
    percentile_lower, percentile_upper = np.percentile(x, q=[percentile, 100 - percentile])
    # print(percentile_lower, percentile_upper)
    x_clipped = np.clip(x, percentile_lower, percentile_upper)

    return x_clipped


# for player_id, player_data_dict in list(data_dict.items())[:1]:
for player_id, player_data_dict in data_dict.items():
    if 'gamelog' not in player_data_dict:
        continue

    # player_id = '9'  ### DEBUG
    # player_data_dict = data_dict[player_id]  ### DEBUG

    player_data_dict_resampled = {}
    df_resampled4player = pd.DataFrame()

    # TODO: think possible missing values in hrm
    # data_source = 'datalog'
    for data_source in player_data_dict.keys():
        if data_source == 'gamelog':
            continue

        if 'time' not in player_data_dict[data_source].columns:
            print('Cant see time in ', data_source)

        df = player_data_dict[data_source]
        df = df.copy()  # DEBUG  # But probably that's allright

        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        if data_source == 'envibox':
            # 'als' column should be dropped here
            df.drop(columns=['mic', 'als'], inplace=True)
            # df['co2'].plot()
            # plt.close()

        if data_source == 'eyetracker':
            df['x_diff'] = np.append(0, np.diff(df['gaze_x'].values))
            df['y_diff'] = np.append(0, np.diff(df['gaze_y'].values))
            df['gaze_movement'] = (df['x_diff'] ** 2 + df['y_diff'] ** 2) ** 0.5
            df.dropna(inplace=True)
            # df['gaze_movement'] = df['gaze_movement'].dropna()
            df['gaze_movement'] = clip_by_percentile(df['gaze_movement'], percentile=5)

            # ###
            # plt.close()
            # plt.scatter(x=df['x_diff'].values, y=df['y_diff'].values)
            # plt.plot(df['gaze_movement'])
            # plt.plot(clip_by_percentile(df['gaze_movement'], percentile=5))
            # plt.close()
            # plt.hist(df['gaze_movement'], bins=100)
            # ###

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
            df['mouse_movement'] = clip_by_percentile(df['mouse_movement'], percentile=5)
            df['mouse_scroll'] = df['mouse_scroll'].abs()
            df.drop(columns=['mouse_dx', 'mouse_dy'], inplace=True)

            # df['mouse_movement'].plot()
            # clip_by_percentile(df['mouse_movement'], 5).plot()
            # plt.close()





        if data_source == 'datalog':
            df.drop(columns=['hrm2'], inplace=True)  # I don't know how to process it yet
            # df.drop(columns=['resistance'], inplace=True)  # Because this data is bullshit. Only 7 players have it correctly measured
            # TODO: consider adding this feature for 7 players only

            # plt.close()
            # df['muscle_activity'].plot()
            serial = df['resistance'] * 1024 / 3.3
            # plt.close()
            # plt.plot(serial)
            serial_max = 768  # I'm just guessing
            # print(serial.max())  # Really need to be checked
            df['resistance'] = (1024 + 2 * serial) * 10000 / (serial_max - serial)
            # plt.plot(df['resistance'].values)
            df['resistance'].dropna(inplace=True)
            # # df['resistance'] = clip_by_percentile(df['resistance'], percentile=0.01)
            # plt.plot(clip_by_percentile(df['resistance'], percentile=0.1).values)
            # plt.savefig(f'pic/features/resistance_player_{player_id}.png')
            # plt.close()

            plt.close()
            # df['muscle_activity'].plot()
            df['muscle_activity'] = (df['muscle_activity'] - df['muscle_activity'].median()).abs()
            # df['muscle_activity'].plot()
            df['muscle_activity'] = clip_by_percentile(df['muscle_activity'], percentile=0.2)
            ### Visualization
            # df['muscle_activity'].plot()
            # clip_by_percentile(df['muscle_activity'], percentile=0.2).plot()


        if data_source == 'schairlog':
            df.drop(columns=['mag_x', 'mag_y', 'mag_z'], inplace=True)
            df = df - df.median()
            df = df.abs()
            df = clip_by_percentile(df, percentile=0.5)

            # col = 'acc_x'
            # df[col].plot()
            # clip_by_percentile(df, percentile=0.5)[col].plot()
            # plt.close()



        # df = df.drop_duplicates()
        unique_values, unique_indexes = np.unique(df.index, return_index=True)
        df = df.iloc[unique_indexes, :]
        # plt.plot(df.index)

        df.rolling(window='100ms').mean()
        if df.isnull().mean().mean() != 0:
            print(f'Portion of na: {df.isnull().mean()}')

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












