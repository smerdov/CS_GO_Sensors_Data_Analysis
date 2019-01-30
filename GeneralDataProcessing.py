import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import normalize_MPU9250_data

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
# data_path = 'data/CSV'
data_path = 'Anonimised Data/Data'

player_folders = os.listdir(data_path)
player_folders = [f"{data_path}/{folder}" for folder in player_folders if not folder.startswith('.')]

data_dict = {}

data_sources_list = ['schairlog', 'gamelog', 'hrm', 'envibox', 'datalog']  # List sources for analysis here

# chair_data_columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']

for player_folder in player_folders:
    player_data_dict = {}
    player_id = player_folder.split('/')[-1]

    player_files = os.listdir(player_folder)
    player_files = [file for file in player_files if not file.startswith('.')]
    player_data_sources = [file.split('_')[0] for file in player_files]  # There are might be repetitions
    # print(player_data_sources)

    for file, data_source in zip(player_files, player_data_sources):
        if data_source not in data_sources_list:
            continue

        try:
            df = pd.read_csv(player_folder + '/' + file)

            if data_source in player_data_dict:  # If already in dict it's appended
                new_df = pd.concat([player_data_dict[data_source], df], axis=0).reset_index(drop=True)
                player_data_dict[data_source] = new_df
            else:
                player_data_dict[data_source] = df
        except:
            pass

    # Sorting by time and fixing naming
    for data_source in player_data_dict.keys():
        if data_source == 'gamelog':
            player_data_dict[data_source].rename(columns={'Unnamed: 0': 'time'}, inplace=True)

        if data_source == 'hrm':
            mask_fake_data = player_data_dict[data_source]['hrm'] < 45
            player_data_dict[data_source] = player_data_dict[data_source].loc[~mask_fake_data, :]

        if data_source == 'datalog':
            player_data_dict[data_source].drop(columns=['time_host', 'n'], inplace=True)
            rename_dict = {
                'sensor1': 'hrm2',
                'sensor2': 'resistance',
                'sensor3': 'muscle_activity',
            }
            player_data_dict[data_source].rename(columns=rename_dict, inplace=True)

        if data_source == 'envibox':
            player_data_dict[data_source].drop(columns=['time_host', 'n'], inplace=True)

        player_data_dict[data_source].sort_values(by='time', inplace=True)
        player_data_dict[data_source].reset_index(drop=True, inplace=True)
        player_data_dict[data_source]['time'] = pd.to_datetime(player_data_dict[data_source]['time']).apply(lambda x: x.timestamp())

    data_dict[player_id] = player_data_dict

joblib.dump(data_dict, 'data/data_dict')


