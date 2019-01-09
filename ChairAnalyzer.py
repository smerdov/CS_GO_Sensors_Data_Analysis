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
                 folder,
                 measurement_interval,
                 pic_prefix,
                 measurements_per_batch=1000,
                 name=None,
                 ):
        self.folder = folder
        self.measurement_interval = measurement_interval
        self.pic_prefix = pic_prefix
        self.measurements_per_batch = measurements_per_batch
        if name is not None:
            self.name = name
        else:
            self.name = folder.split('/')[-1]

        self.get_df_total()

    def get_df_total(self):
        folder = self.folder

        filenames_list = os.listdir(folder)
        filenames_list = sorted([int(x) for x in filenames_list])
        filenames_list = [str(x) for x in filenames_list]

        df_total = None

        for filename in filenames_list:
            print(filename)

            # dicts_list = joblib.load(folder + '/' + filename)
            dicts_list = []
            with open(folder + '/' + filename) as f:
                lines = f.readlines()
                # print(len(lines))
                if len(lines) == 0:
                    continue

                for line in lines:
                    try:
                        new_dict = json.loads(line)
                        new_dict['datetime_now'] = self.parse_string_iso_format(new_dict['datetime_now'])
                        dicts_list.append(new_dict)
                    except:
                        break

            df2append = pd.DataFrame(dicts_list)

            if df_total is None:
                df_total = df2append
            else:
                df_total = pd.concat([df_total, df2append], axis=0)

        rename_dict = {
            'accelerometer_x': 'Acc_x',
            'accelerometer_y': 'Acc_y',
            'accelerometer_z': 'Acc_z',
            'magnetometer_x': 'Mag_x',
            'magnetometer_y': 'Mag_y',
            'magnetometer_z': 'Mag_z',
            b'accelerometer_x': 'Acc_x',
            b'accelerometer_y': 'Acc_y',
            b'accelerometer_z': 'Acc_z',
            b'magnetometer_x': 'Mag_x',
            b'magnetometer_y': 'Mag_y',
            b'magnetometer_z': 'Mag_z',
        }
        if df_total is not None:
            df_total.rename(columns=rename_dict, inplace=True)
            df_total.reset_index(inplace=True, drop=True)

        self.df_total = df_total

    # df_total = get_df_total(folder='ivan_0')
    def plot_measurements_timeline(
            self,
            sensors=('accel', 'gyro', 'mag'),
            axes=('x', 'y', 'z'),
            # filename='measurements_timeline',
    ):
        df = self.df_total
        name = self.name
        measurement_interval = self.measurement_interval

        n_cols = len(sensors)
        n_rows = len(axes)

        fig, ax = plt.subplots(n_rows, n_cols, sharex='col', figsize=(19, 11))

        for n_row, n_col in itertools.product(range(n_rows), range(n_cols)):
            ax_instance = ax[n_row, n_col]

            column_name = sensors[n_col] + '_' + axes[n_row]
            data2plot = df.loc[:, column_name].values

            ax_instance.plot(data2plot)

            if n_row == 0:
                title = sensors[n_col]
                ax_instance.set_title(title)

            if n_col == 0:
                title = axes[n_row]
                ax_instance.set_ylabel(title)

        zeros_portions = self.get_zeros_portion()
        mag_zeros_portion = zeros_portions[['mag_x', 'mag_y', 'mag_z']].mean()
        mag_zeros_string = f'Mag zeros portion = {round(mag_zeros_portion, 3)}'

        suptitle = f'measurement_interval = {measurement_interval}, ' + mag_zeros_string
        plt.suptitle(suptitle)
        # plt.tight_layout(rect=[0, 0, 1, 0.5])
        fig.tight_layout(rect=[0, 0.00, 1, 0.97])
        # fig.subplots_adjust(top=0.85)
        # plt.savefig(pic_prefix + filename)
        plt.savefig(pic_prefix + f'measurements_timeline_{name}.png')
        plt.close()

    # create means_stds ?

    # def get_lean_back_portion(acc_z, means_stds=means_stds, n_sigma=5):
    def get_lean_back_portion(self, acc_z, acc_z_mean=-15910, acc_z_std=30, n_sigma=3):
        #     result = {}
        # acc_z_mean = means_stds.loc['Acc_z', 'mean']
        # acc_z_std = means_stds.loc['Acc_z', 'std']

        acc_z_min = acc_z_mean - n_sigma * acc_z_std
        acc_z_max = acc_z_mean + n_sigma * acc_z_std

        lean_back_portion = (acc_z < acc_z_min).mean()
        #     result['lean_back_portion'] = lean_back_portion

        #     return result
        return lean_back_portion

    def get_mess_mask_acc(self, acc_data, percentile2crop=10, n_sigma=10):
        lower_bound, upper_bound, median = np.percentile(acc_data, [percentile2crop, 100 - percentile2crop, 50])
        acc_data_filtered = acc_data[(lower_bound < acc_data) & (acc_data < upper_bound)]
        std = np.std(acc_data_filtered)
        oscillation = std / (25 * n_sigma)

        # Calculating bound for calm state
        calm_state_lower_bound = median - n_sigma * std
        calm_state_upper_bound = median + n_sigma * std

        mask_calm = ((calm_state_lower_bound < acc_data) & (acc_data < calm_state_upper_bound)).values
        #     mess_portion = 1 - np.mean(mask_calm)

        #     return mess_portion
        return mask_calm, oscillation

    def get_mess_mask_mag(self, mag_data, w=0.05, max_calm_derivative=30):
        # Spline approximation
        y = mag_data.values
        x = np.arange(len(y))
        splines = splrep(x, y, w=w * np.ones_like(y))
        points = splev(x, splines, der=0)
        derivatives = splev(x, splines, der=1)

        mask_calm = abs(derivatives) < max_calm_derivative

        #     return points, derivatives
        return mask_calm

    def get_mess_mask_mag4graph(self, mag_data, w=0.05, max_calm_derivative=30):
        # Spline approximation
        y = mag_data.values
        x = np.arange(len(y))
        splines = splrep(x, y, w=w * np.ones_like(y))
        points = splev(x, splines, der=0)
        derivatives = splev(x, splines, der=1)

        mask_calm = abs(derivatives) < max_calm_derivative

        return points, derivatives

    def get_chair_stats(self):
        df_total = self.df_total
        # results_list = []

        mask_calm_acc_x, oscillation_acc_x = self.get_mess_mask_acc(df_total['Acc_x'])
        mask_calm_acc_y, oscillation_acc_y = self.get_mess_mask_acc(df_total['Acc_y'])
        mask_calm_acc_z, oscillation_acc_z = self.get_mess_mask_acc(df_total['Acc_z'])

        mess_portion_acc_x = 1 - mask_calm_acc_x.mean()
        mess_portion_acc_y = 1 - mask_calm_acc_y.mean()
        mess_portion_acc_z = 1 - mask_calm_acc_z.mean()

        mess_portion_acc = (oscillation_acc_x + oscillation_acc_y + oscillation_acc_z) / 3

        mask_calm_acc = mask_calm_acc_x & mask_calm_acc_y & mask_calm_acc_z
        mess_portion_acc = 1 - mask_calm_acc.mean()

        mask_calm_mag_x = self.get_mess_mask_mag(df_total['Mag_x'])
        mask_calm_mag_y = self.get_mess_mask_mag(df_total['Mag_y'])
        mask_calm_mag_z = self.get_mess_mask_mag(df_total['Mag_z'])

        mess_portion_mag_x = 1 - mask_calm_mag_x.mean()
        mess_portion_mag_y = 1 - mask_calm_mag_y.mean()
        mess_portion_mag_z = 1 - mask_calm_mag_z.mean()

        mask_calm_mag = mask_calm_mag_x & mask_calm_mag_y & mask_calm_mag_z
        mess_portion_mag = 1 - mask_calm_mag.mean()

        lean_back_portion = self.get_lean_back_portion(df_total['Acc_z'])
        result = {
            # 'people_id': people_id,
            'mess_portion_acc_x': mess_portion_acc_x,
            'mess_portion_acc_y': mess_portion_acc_y,
            'mess_portion_acc_z': mess_portion_acc_z,
            'mess_portion_acc': mess_portion_acc,
            'lean_back_portion': lean_back_portion,
            'mess_portion_mag_x': mess_portion_mag_x,
            'mess_portion_mag_y': mess_portion_mag_y,
            'mess_portion_mag_z': mess_portion_mag_z,
            'mess_portion_mag': mess_portion_mag,
            'oscillation_acc_x': oscillation_acc_x,
            'oscillation_acc_y': oscillation_acc_y,
            'oscillation_acc_z': oscillation_acc_z,
            'oscillation_acc': oscillation_acc_z,
            # 'stress': stress,
        }
        # results_list.append(result)

        # return results_list
        return result

    def get_chair_stats_truncated(self):
        # self.get_df_total()
        self.plot_measurements_timeline()
        chair_stats_detailed = self.get_chair_stats()

        rename_dict = {
            'mess_portion_acc': 'Momentum',
            'mess_portion_mag': 'Rotational movement',
            'lean_back_portion': 'Lean back',
            'oscillation_acc': 'Oscillation',
        }

        chair_stats_detailed_truncated = {rename_dict[key]: chair_stats_detailed[key] for key in rename_dict if
                                          key in rename_dict}

        return chair_stats_detailed_truncated

    def plot_measurement_times(self):  # , filename='time_wrt_step.png'):
        df = self.df_total
        pic_prefix = self.pic_prefix
        measurement_interval = self.measurement_interval
        measurements_per_batch = self.measurements_per_batch
        n_measurements = len(df)
        n_batches = n_measurements // self.measurements_per_batch
        name = self.name

        timestamp_start = df['datetime_now'].min().timestamp()
        time_passed = df['datetime_now'].apply(lambda x: x.timestamp() - timestamp_start)

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
        df = self.df_total.drop('datetime_now', axis=1)
        zeros_portions = (df == 0).mean(axis=0)

        return zeros_portions

    @staticmethod
    def parse_string_iso_format(s):
        d = dateutil.parser.parse(s)
        return d
