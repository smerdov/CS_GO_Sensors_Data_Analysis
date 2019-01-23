import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import normalize_MPU9250_data, split_df
from ChairAnalyzer import ChairAnalyser

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
# data_path = 'data/CSV'
# data_path = 'Anonimised Data/Data'

data_dict = joblib.load('data/data_dict')

chair_features_list = []

def get_chair_features(df_chair, player_id):
    chair_analyser = ChairAnalyser(df_chair, 0.01, pic_prefix, name=player_id)  # + f'_{n_chunk}')
    nonstationary_values_portion = chair_analyser.get_nonstationary_values_portion()
    lean_back_portion = chair_analyser.get_lean_back_portion()
    oscillations = chair_analyser.get_oscillation_intensity()

    chair_features = pd.concat([nonstationary_values_portion, lean_back_portion, oscillations])

    return chair_features


for player_id, player_data_dict in data_dict.items():
    if 'schairlog' in player_data_dict:
        df_chair = player_data_dict['schairlog']
    else:
        continue

    ### Each chair log is splitted to small sessions
    chunk_lenght = 100 * 600
    df_chunks_list = split_df(df_chair, n_chunks=3, chunk_lenght=chunk_lenght)
    # print(len(df_chunks_list))
    # chair_analyser = ChairAnalyser(df_chair, 0.01, pic_prefix, name=player_id)  # + f'_{n_chunk}')
    # chair_analyser.plot_measurements_timeline(sensors=('acc', 'gyro'), plot_suptitle=False, fontsize=22)

    for n_chunk, df_chunk in enumerate(df_chunks_list):
        chair_features = get_chair_features(df_chunk, player_id)
        chair_features_list.append(chair_features)


df_chair_features = pd.DataFrame(chair_features_list)
df_chair_features.reset_index(inplace=True)
df_chair_features.rename(columns={'index': 'player_id'}, inplace=True)

df_chair_features.to_csv('data/chair_features.csv', index=False)