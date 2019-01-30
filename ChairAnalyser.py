import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GeneralAnalyser import GeneralAnalyser


class ChairAnalyser(GeneralAnalyser):

    def __init__(self,
                 df,
                 pic_prefix,
                 sensor_name,
                 session_id,
                 measurement_interval=0.01,
                 measurements_per_batch=1000,
                 name=None,
                 ):
        super().__init__(df, pic_prefix, sensor_name, session_id)

        # self.df_chair = df_chair
        self.measurement_interval = measurement_interval
        self.pic_prefix = pic_prefix
        self.measurements_per_batch = measurements_per_batch
        self.name = name

        self.means, self.stds, medians = self.create_mean_stds()

    def create_mean_stds(self, columns=('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')):
        df_chair = self.df.loc[:, columns]
        # df_chair = df_chair.loc[:, columns]
        # medians, lower_bounds, upper_bounds = np.percentile(df_chair, [50, percentile2crop, 100 - percentile2crop], axis=0)

        means = df_chair.mean(axis=0)
        medians = df_chair.median(axis=0)
        stds = df_chair.std(axis=0)

        return means, stds, medians

    def get_nonstationary_values_portion(self, n_sigma=3):
        means = self.means
        stds = self.stds

        columns = stds.index
        df_chair = self.df.loc[:, columns]

        lower_bounds = means - n_sigma * stds
        upper_bounds = means + n_sigma * stds

        low_values_means = (df_chair.loc[:, columns] < lower_bounds).mean()
        high_values_means = (df_chair.loc[:, columns] > upper_bounds).mean()

        nonstationary_values_portion = low_values_means + high_values_means
        nonstationary_values_portion.index = [colname + '__nonstationary_portion' for colname in nonstationary_values_portion.index]
        nonstationary_values_portion.name = self.name

        return nonstationary_values_portion

    # def get_lean_back_portion(acc_z, means_stds=means_stds, n_sigma=5):
    def get_lean_back_portion(self, acc_z_threshold=0.97):
        df_chair = self.df
        lean_back_portion = (df_chair[['acc_z']] < acc_z_threshold).mean()
        lean_back_portion.index = ['lean_back_portion']
        lean_back_portion.name = self.name

        return lean_back_portion

    def get_oscillation_intensity(self, percentile2crop=10, columns=('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')):
        df_chair = self.df.loc[:, columns]
        result = {}

        for column in columns:
            lower_bounds, upper_bounds = np.percentile(df_chair.loc[:, column], [percentile2crop, 100 - percentile2crop], axis=0)
            # intervals = upper_bounds - lower_bounds
            low_values_mask = (df_chair.loc[:, column] < lower_bounds)
            high_values_mask = (df_chair.loc[:, column] > upper_bounds)

            normal_values_mask = (~low_values_mask) & (~high_values_mask)

            usual_sitting_stds = df_chair.loc[normal_values_mask, column].std()
            oscillations = usual_sitting_stds# / intervals
            feature_name = f'{column}__oscillations'
            result[feature_name] = oscillations

        result = pd.Series(result)
        result.name = self.name

        return result







