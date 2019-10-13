import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
from utils import string2json
from config import TIMESTEP
import itertools
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

plt.interactive(True)
pd.options.display.max_columns = 15
pic_folder = 'pic/'

data_dict_resampled_merged_with_target = joblib.load('data/data_dict_resampled_merged_with_target')

# df_all = pd.DataFrame()

df_all = pd.concat(list(data_dict_resampled_merged_with_target.values()), axis=0)
target_prefix = 'kills_proportion'
target_columns = [column for column in df_all.columns if column.startswith(target_prefix)]
columns_order = sorted(df_all.drop(columns=target_columns).columns)
# columns_order = ['als', 'co2', 'hrm', 'hrm2', 'humidity', 'mic',
#                  'muscle_activity', 'resistance', 'temperature']
# columns_order.remove(target_column)
# df_all.isnull().mean()

# for df in data_dict_resampled_merged_with_target.values():
#     print(df.isnull().mean())

ss = StandardScaler()
ss.fit(df_all.loc[:, columns_order])
joblib.dump(ss, 'data/ss_0')
joblib.dump(ss, 'data/columns_order_0')

# ss.inverse_transform(df_all.values)

data_dict_resampled_merged_with_target_scaled = {}

for player_id, df in data_dict_resampled_merged_with_target.items():
    for column in columns_order:
        if column not in df:
            df[column] = np.nan

    df_scaled = df.loc[:, columns_order]
    df_scaled.loc[:, columns_order] = ss.transform(df_scaled.loc[:, columns_order])
    df_scaled.fillna(0, inplace=True)

    for target_column in target_columns:
        df_scaled[target_column] = df[target_column]

    # df.loc[:, columns_order] = ss.transform(df.loc[:, columns_order])
    data_dict_resampled_merged_with_target_scaled[player_id] = df_scaled


joblib.dump(data_dict_resampled_merged_with_target_scaled, 'data/data_dict_resampled_merged_with_target_scaled')


# splitter = KFold(n_splits=5, shuffle=True)
#
# split = splitter.split(player_ids)
#
# # for split_train, split_val in splitter.split(player_ids):
# for x in splitter.split(player_ids):
#     print(x)
#
# sklearn.model_selection












