import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import normalize_MPU9250_data, split_df, get_intervals_from_moments, EventIntervals
from GeneralAnalyser import GeneralAnalyser, plot_measurements

# plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
# data_path = 'data/CSV'
# data_path = 'Anonimised Data/Data'

# sessions_dict = joblib.load('data/sessions_dict')
sessions_dict = joblib.load('data/sessions_dict')
gamedata_dict = joblib.load('data/gamedata_dict')

sensors_columns_dict = {
    'hrm': ['hrm'],
    'envibox': ['als', 'mic', 'humidity', 'temperature', 'co2'],
    'datalog': ['hrm2', 'resistance', 'muscle_activity']
}

sensors_list = list(sensors_columns_dict.keys())
sensors_columns_list = []

for session_id, session_data_dict in sessions_dict.items():
    df_dict = {}

    if not set(sensors_list).issubset(set(session_data_dict.keys())):
        continue

    if session_id not in gamedata_dict:
        continue

    df_discretized_list = []

    for sensor_name in sensors_columns_dict:
        df = session_data_dict[sensor_name]
        df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df['time'], unit='s')))
        df_discretized = df.resample('100ms').mean().ffill()  # Forward fill is better
        df_discretized_list.append(df_discretized)

        moments_kills = gamedata_dict[session_id]['times_kills']
        moments_death = gamedata_dict[session_id]['times_is_killed']
        duration = 1
        intervals_shootout = gamedata_dict[session_id]['shootout_times_start_end']
        intervals_kills = get_intervals_from_moments(moments_kills, interval_start=-duration, interval_end=duration)
        intervals_death = get_intervals_from_moments(moments_death, interval_start=-duration, interval_end=duration)

        event_intervals_shootout = EventIntervals(intervals_list=intervals_shootout, label='shootouts', color='blue')
        event_intervals_kills = EventIntervals(intervals_list=intervals_kills, label='kills', color='green')
        event_intervals_death = EventIntervals(intervals_list=intervals_death, label='deaths', color='red')

def discretize_time_column(time_column, discretization=0.1):
    time_column_discretized = time_column - time_column % discretization
    return time_column_discretized


def auxilary_discretization_table(time_column, discretization):
    time_column_discretized = discretize_time_column(df['time'], discretization)
    timesteps = np.arange(0, time_column_discretized.max() + discretization, discretization)
    '''
    If there are several records to one timestep => select the earliest
    If there isn't any records to one timestep => select the latest available
    '''

pd.PeriodIndex(df['time'])
pd.TimedeltaIndex(df['time'])
df = df.set_index()
nano = df.resample('1ns')
nano.sample()

event_intervals_shootout.intervals_list
time_column_discretized = df['time'] - df['time'] % 0.1
time_column_discretized = pd.Series(np.arange(0, 180, 0.1))
game_mask = event_intervals_shootout.get_mask_intervals_union(time_column_discretized)
game_mask = 1 * game_mask




df_merged = pd.concat(df_discretized_list, axis=1)
df_merged = df_merged.ffill().bfill()

df_merged = df_merged.drop(['time'], axis=1)
df_merged = df_merged.reset_index(drop=True)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
train = ss.fit_transform(df_merged)









import torch
import torch.nn as nn
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence

input_size = train.shape[1]
hidden_size = input_size


lstm = nn.LSTM(input_size, hidden_size)
opt = Adam(lstm.parameters())

from torch.utils.data import TensorDataset, DataLoader

train = torch.Tensor(train)
target = torch.Tensor(game_mask).long()

dataset = TensorDataset(train, target)
data_loader = DataLoader(dataset, batch_size=8)

pack_padded_sequence(train)
list(pack_sequence(train))[0].shape






for x_batch, y_batch in data_loader:
    lstm(x_batch)
































