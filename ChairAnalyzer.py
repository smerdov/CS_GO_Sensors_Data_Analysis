import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import itertools
from scipy.interpolate import splev, splrep
import json
from datetime import datetime
import dateutil.parser

pic_prefix = '../../pic/'

class ChairAnalyser:

    def __init__(self,
                 df,
                 measurement_interval,
                 pic_prefix,
                 measurements_per_batch=1000,
                 name=None,
                 ):
        self.df_total = df
        self.measurement_interval = measurement_interval
        self.pic_prefix = pic_prefix
        self.measurements_per_batch = measurements_per_batch
        self.name = name

        self.means, self.stds, medians = self.create_mean_stds()


    def plot_measurements_timeline(
            self,
            sensors=('acc', 'gyro', 'mag'),
            axes=('x', 'y', 'z'),
            plot_suptitle=True,
            fontsize=18,
    ):
        df = self.df_total
        name = self.name
        measurement_interval = self.measurement_interval

        n_cols = len(sensors)
        n_rows = len(axes)

        fig, ax = plt.subplots(n_rows, n_cols, sharex='col', figsize=(14, 9.5))


        for n_row, n_col in itertools.product(range(n_rows), range(n_cols)):
            ax_instance = ax[n_row, n_col]

            column_name = sensors[n_col] + '_' + axes[n_row]
            data2plot = df.loc[:, column_name].values

            ax_instance.plot(data2plot)
            # plt.xticks(fontsize=fontsize - 2)
            # plt.yticks(fontsize=fontsize - 2)

            if n_row == 0:
                title = sensors[n_col]
                if title == 'acc':
                    title = 'Accelerometer'
                elif title == 'gyro':
                    title = 'Gyroscope'
                ax_instance.set_title(title, fontsize=fontsize)

            if n_col == 0:
                title = axes[n_row]
                ax_instance.set_ylabel(title, fontsize=fontsize)

        if plot_suptitle:
            suptitle = f'measurement_interval = {measurement_interval}'

        if 'mag' in sensors:
            zeros_portions = self.get_zeros_portion()
            mag_zeros_portion = zeros_portions[['mag_x', 'mag_y', 'mag_z']].mean()
            if plot_suptitle:
                mag_zeros_string = f'Mag zeros portion = {round(mag_zeros_portion, 3)}'
                suptitle = suptitle + ', ' + mag_zeros_string

        if plot_suptitle:
            plt.suptitle(suptitle, fontsize=fontsize + 2)

        fig.tight_layout(rect=[0, 0.00, 1, 0.97])
        plt.savefig(pic_prefix + f'measurements_timeline_{name}.png')
        plt.close()

    def create_mean_stds(self, columns=('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')):
        df_chair = self.df_total.loc[:, columns]
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
        df_chair = self.df_total.loc[:, columns]

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
        df_chair = self.df_total
        lean_back_portion = (df_chair[['acc_z']] < acc_z_threshold).mean()
        lean_back_portion.index = ['lean_back_portion']
        lean_back_portion.name = self.name

        return lean_back_portion

    def get_oscillation_intensity(self, percentile2crop=10, columns=('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')):
        df_chair = self.df_total.loc[:, columns]
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

    def plot_measurement_times(self):  # , filename='time_wrt_step.png'):
        df = self.df_total
        pic_prefix = self.pic_prefix
        measurement_interval = self.measurement_interval
        measurements_per_batch = self.measurements_per_batch
        n_measurements = len(df)
        n_batches = n_measurements // self.measurements_per_batch
        name = self.name

        timestamp_start = df['time'].min().timestamp()
        time_passed = df['time'].apply(lambda x: x.timestamp() - timestamp_start)

        # index2drop = range(measurements_per_batch, n_measurements, measurements_per_batch)
        # time_passed_truncated = time_passed.drop(index2drop, axis=0)

        time_between_batches_array = time_passed[measurements_per_batch::measurements_per_batch].values - \
                                     time_passed[measurements_per_batch - 1:-1:measurements_per_batch].values
        time_between_batches = time_between_batches_array.mean()

        timediff_total = time_passed.iloc[-1]
        timediff_because_of_measurements = timediff_total - time_between_batches_array.sum()
        n_measurements_without_batch = n_measurements - n_batches
        time_between_measurements = timediff_because_of_measurements / n_measurements_without_batch

        plt.close()
        plt.figure(figsize=(16, 12))
        plt.plot(time_passed)
        plt.xlabel('n_step')
        plt.ylabel('Time passed, s')
        title = f'Measurement interval = {round(measurement_interval, 3)}, ' + \
                f'Time Between Measurements = {round(time_between_measurements, 3)}, ' + \
                f'Time Between Batches = {round(time_between_batches, 3)}'
        plt.title(title, fontsize=16)
        plt.tight_layout()
        # plt.savefig(pic_prefix + filename)
        plt.savefig(pic_prefix + f'time_wrt_step_{name}.png')

    def get_zeros_portion(self):
        df = self.df_total.drop('time', axis=1)
        zeros_portions = (df == 0).mean(axis=0)

        return zeros_portions

    @staticmethod
    def parse_string_iso_format(s):
        d = dateutil.parser.parse(s)
        return d



