import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from utils import normalize_MPU9250_data

plt.interactive(True)
pd.options.display.max_columns = 15
pic_prefix = 'pic/'
data_path = 'data/CSV'

folders = os.listdir(data_path)
folders = [f"{data_path}/{folder}" for folder in folders if not folder.startswith('.')]

data_dict_dict = {}

chair_data_columns = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']

for folder in folders:
    data_dict = {}
    name = folder.split('/')[-1]

    files = os.listdir(folder)
    files_shortnames = [file.split('_')[0] for file in files]  # There are might be repetitions

    for file, file_shortname in zip(files, files_shortnames):
        if file_shortname != 'schairlog':
            continue

        try:
            df = pd.read_csv(folder + '/' + file)

            if file_shortname in data_dict:  # If already in dict it's appended
                new_df = pd.concat([data_dict[file_shortname], df], axis=0).reset_index(drop=True)
                data_dict[file_shortname] = new_df
            else:
                data_dict[file_shortname] = df
        except:
            pass

    data_dict_dict[name] = data_dict

chair_data_dict = {}






keys = list(data_dict_dict.keys())

keys

























