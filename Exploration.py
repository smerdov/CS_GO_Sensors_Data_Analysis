import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import normalize_MPU9250_data, split_df
from ChairAnalyzer import ChairAnalyser

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
data_path = 'data/CSV'

folders = os.listdir(data_path)
folders = [f"{data_path}/{folder}" for folder in folders if not folder.startswith('.')]

data_dict_dict = {}

chair_data_columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']

data_sources_list = ['schairlog']  # List sources for analysis here

for folder in folders:
    data_dict = {}
    name = folder.split('/')[-1]

    files = os.listdir(folder)
    files = [file for file in files if not file.startswith('.')]
    data_sources = [file.split('_')[0] for file in files]  # There are might be repetitions
    print(data_sources)

    for file, data_source in zip(files, data_sources):
        if data_source not in data_sources_list:
            continue

        try:
            df = pd.read_csv(folder + '/' + file)

            if data_source in data_dict:  # If already in dict it's appended
                new_df = pd.concat([data_dict[data_source], df], axis=0).reset_index(drop=True)
                data_dict[data_source] = new_df
            else:
                data_dict[data_source] = df
        except:
            pass

    data_dict_dict[name] = data_dict

chair_data_dict = {}

for key, value in data_dict_dict.items():
    key = key.replace('\t', ' ')
    if 'schairlog' in value:
        df_chair = value['schairlog']
        chair_data_dict[key] = df_chair
        print(len(df_chair))

# keys = list(data_dict_dict.keys())
# data_dict_dict[keys[0]]['schairlog']


nonstationary_values_portion_list = []
# TODO: do not draw pictures

for player_name, df_chair in chair_data_dict.items():
    chunk_lenght = 100 * 300
    df_chunks_list = split_df(df_chair, n_chunks=10, chunk_lenght=chunk_lenght)
    print(len(df_chunks_list))
    # chair_analyser = ChairAnalyser(df_chair, 0.01, pic_prefix, name=player_name)  # + f'_{n_chunk}')
    # chair_analyser.plot_measurements_timeline(sensors=('acc', 'gyro'), plot_suptitle=False, fontsize=22)

    for n_chunk, df_chunk in enumerate(df_chunks_list):
        chair_analyser = ChairAnalyser(df_chunk, 0.01, pic_prefix, name=player_name)# + f'_{n_chunk}')
        # chair_analyser.plot_measurements_timeline(sensors=('acc', 'gyro'))
        nonstationary_values_portion = chair_analyser.get_nonstationary_values_portion()
        lean_back_portion = chair_analyser.get_lean_back_portion()
        oscillations = chair_analyser.get_oscillation_intensity()

        nonstationary_values_portion = nonstationary_values_portion.append(lean_back_portion)
        nonstationary_values_portion = nonstationary_values_portion.append(oscillations)

        nonstationary_values_portion_list.append(nonstationary_values_portion)


df_nonstationary_values_portion = pd.DataFrame(nonstationary_values_portion_list)
df_nonstationary_values_portion.reset_index(inplace=True)
df_nonstationary_values_portion.rename(columns={'index': 'player_name'}, inplace=True)

df_players = pd.read_csv('../data/participants2_fixed.csv', sep=';')
df_players['player_name'] = df_players[['First Name', 'Last Name']].apply(lambda x: ' '.join(x), axis=1)

df_players.rename(columns={
    ' What experience do u have in shooter games (Counter-Strike, Doom, Battlefield, etc.)?': 'Skill'
    },
    inplace=True,
)

df_players = df_players[['player_name', 'Skill']]
skill_is_none = df_players['Skill'] == 'None'
df_players.loc[skill_is_none, 'Skill'] = 'Small'




























