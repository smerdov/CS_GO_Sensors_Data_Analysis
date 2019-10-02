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
from collections import OrderedDict
from utils import get_aspect_from_n_plots
from sklearn.preprocessing import StandardScaler

######## Concept
# One function for visualisation which can receive multiple sources. The function has "structured" mode for visualisation.
# GeneralAnalyser for general data processing.
# Child for each data_source.
# Scatter plot for sensor-sensor interactions
########

class GeneralAnalyser:

    def __init__(self,
                 df,
                 pic_prefix,
                 sensor_name,
                 session_id,
                 name=None,
                 ):
        self.df = df
        self.pic_prefix = pic_prefix
        self.sensor_name = sensor_name
        self.session_id = session_id
        self.name = name  # Need to be elaborated

    # def plot_measurements_timeline(
    #         self,
    #         column_name,
    #         intervals_dicts_list=None,
    #         plot_suptitle=False,
    #         fontsize=18,
    #         alpha=0.9,
    #         alpha_background=0.5,
    # ):
    #     df = self.df
    #
    #     times = df['time']
    #
    #     fig, ax_instance = plt.subplots(1, 1, sharex='col', figsize=(14, 9.5))
    #
    #     # column_name = sensors[n_col] + '_' + axes[n_row]
    #     data2plot = df.loc[:, column_name]
    #
    #     ax_instance.plot(times, data2plot.values, label='nothing', color='black', alpha=alpha_background)
    #
    #     for intervals_dict in intervals_dicts_list:
    #         mask_interval_list = get_mask_intervals(df['time'], intervals_list=intervals_dict['intervals_list'])
    #         label = intervals_dict['label']
    #         color = intervals_dict['color']
    #
    #         for mask_interval in mask_interval_list:
    #             times_with_mask = times.loc[mask_interval]
    #             data2plot_with_mask = data2plot.loc[mask_interval]
    #             ax_instance.plot(
    #                 times_with_mask,
    #                 data2plot_with_mask.values,
    #                 # label=label,
    #                 color=color,
    #                 alpha=alpha,
    #             )
    #
    #         ax_instance.plot([], [], label=label, color=color)
    #     ax_instance.legend(loc='upper right')
    #
    #     # if n_row == 0:
    #     #     title = sensors[n_col]
    #     #     if title == 'acc':
    #     #         title = 'Accelerometer'
    #     #     elif title == 'gyro':
    #     #         title = 'Gyroscope'
    #     #     ax_instance.set_title(title, fontsize=fontsize)
    #     #
    #     # if n_col == 0:
    #     #     title = axes[n_row]
    #     #     ax_instance.set_ylabel(title, fontsize=fontsize)
    #     #
    #     # if plot_suptitle:
    #     #     suptitle = f'measurement_interval = {measurement_interval}'
    #     #
    #     # if 'mag' in sensors:
    #     #     zeros_portions = self.get_zeros_portion()
    #     #     mag_zeros_portion = zeros_portions[['mag_x', 'mag_y', 'mag_z']].mean()
    #     #     if plot_suptitle:
    #     #         mag_zeros_string = f'Mag zeros portion = {round(mag_zeros_portion, 3)}'
    #     #         suptitle = suptitle + ', ' + mag_zeros_string
    #     #
    #     # if plot_suptitle:
    #     #     plt.suptitle(suptitle, fontsize=fontsize + 2)
    #
    #     fig.tight_layout(rect=[0, 0.00, 1, 0.97])
    #     plt.savefig(self.pic_prefix + f'measurements_{self.sensor_name}_{self.session_id}.png')
    #     plt.close()


def plot_measurements_pairwise(
        analyser_column_pairs_pairs_list,
        pic_prefix,
        session_id,
        event_intervals_list=None,
        n_rows=None,
        n_cols=None,
        figsize=(21, 15),
        plot_suptitle=False,
        fontsize=18,
        alpha=0.9,
        alpha_background=0.5,
        point_size=2,
        sharex='col',
):
    n_plots = len(analyser_column_pairs_pairs_list)

    if n_rows is None:  # TODO: can be isolated to a function
        if n_cols is None:
            n_rows, n_cols = get_aspect_from_n_plots(n_plots)
        else:
            n_rows = int(np.ceil(n_plots / n_cols))
    else:
        if n_cols is None:
            n_cols = int(np.ceil(n_plots / n_rows))

    # analysers_names = [analyser.sensor_name for analyser, column in analyser_column_pairs_list]
    # analysers_names = list(OrderedDict.fromkeys(analysers_names))  # To preserve uniqueness and order
    # pic_path = pic_prefix + 'measurements_' + '_'.join(analysers_names) + f'_{session_id}' + '.png'
    pic_path = pic_prefix + 'measurements_' + 'pairwise' + f'_{session_id}' + '.png'

    fig, ax_list = plt.subplots(n_rows, n_cols, sharex=sharex, figsize=figsize, squeeze=False)
    rows_cols_list = itertools.product(range(n_rows), range(n_cols))

    for analyser_column_pair_pair, row_col_pair in zip(analyser_column_pairs_pairs_list, rows_cols_list):
        analyser_column_pair_1, analyser_column_pair_2 = analyser_column_pair_pair
        analyser_1, column_1 = analyser_column_pair_1
        analyser_2, column_2 = analyser_column_pair_2
        n_row, n_col = row_col_pair
        ax = ax_list[n_row, n_col]

        times = analyser_1.df['time']
        # data2plot = analyser.df[column]
        data_1 = analyser_1.df[column_1]
        data_2 = analyser_2.df[column_2]

        # Normalizing
        ss_1 = StandardScaler()
        ss_2 = StandardScaler()
        data_1.loc[:] = ss_1.fit_transform(data_1.values.reshape(-1, 1)).ravel()
        data_2.loc[:] = ss_2.fit_transform(data_2.values.reshape(-1, 1)).ravel()

        ax.scatter(data_1.values, data_2.values, label='nothing', color='black', alpha=alpha_background, s=point_size)
        ax.set_xlabel(column_1)
        ax.set_ylabel(column_2)

        for event_intervals in event_intervals_list:
            # intervals_list = event_intervals.intervals_list
            event_label = event_intervals.label
            color = event_intervals.color
            # mask_interval_list = get_mask_intervals(times, intervals_list=intervals_list)
            mask_interval_list = event_intervals.get_mask_intervals(times)

            for mask_interval in mask_interval_list:
                data_1_with_mask = data_1.loc[mask_interval]
                data_2_with_mask = data_2.loc[mask_interval]
                ax.scatter(
                    data_1_with_mask.values,
                    data_2_with_mask.values,
                    # label=event_label,
                    color=color,
                    alpha=alpha,
                    s=point_size,
                )

            ax.plot([], [], label=event_label, color=color)
        ax.legend(loc='upper right')

    if plot_suptitle:  # TODO: deal with suptitle
        suptitle = f'session_id = {session_id}'
        fig.suptitle(suptitle, fontsize=fontsize + 2)

    fig.tight_layout(rect=[0, 0.00, 1, 0.97])
    # fig.tight_layout()

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.subplots_adjust(top=0.5)
    # fig.tight_layout()
    # plt.savefig(pic_prefix + f'measurements_{self.sensor_name}_{self.session_id}.png')
    plt.savefig(pic_path)
    plt.close()




def plot_measurements(
        analyser_column_pairs_list,  # analysers for hrm, temperature, etc. for the same session_id
        pic_prefix,
        session_id,
        event_intervals_list=None,
        n_rows=None,
        n_cols=None,
        figsize=(21, 15),
        plot_suptitle=False,
        fontsize=18,
        alpha=0.9,
        alpha_background=0.5,
        sharex='col',
):
    n_plots = len(analyser_column_pairs_list)

    if n_rows is None:  # TODO: can be isolated to a function
        if n_cols is None:
            n_rows, n_cols = get_aspect_from_n_plots(n_plots)
        else:
            n_rows = int(np.ceil(n_plots / n_cols))
    else:
        if n_cols is None:
            n_cols = int(np.ceil(n_plots / n_rows))

    analysers_names = [analyser.sensor_name for analyser, column in analyser_column_pairs_list]
    analysers_names = list(OrderedDict.fromkeys(analysers_names))  # To preserve uniqueness and order
    pic_path = pic_prefix + 'measurements_' + '_'.join(analysers_names) + f'_{session_id}' + '.png'

    fig, ax_list = plt.subplots(n_rows, n_cols, sharex=sharex, figsize=figsize, squeeze=False)
    rows_cols_list = itertools.product(range(n_rows), range(n_cols))

    for analyser_column_pair, row_col_pair in zip(analyser_column_pairs_list, rows_cols_list):
        analyser, column = analyser_column_pair
        n_row, n_col = row_col_pair
        ax = ax_list[n_row, n_col]

        times = analyser.df['time']
        data2plot = analyser.df[column]
        sensor_name = analyser.sensor_name

        ax.plot(times, data2plot.values, label='nothing', color='black', alpha=alpha_background)
        ax.tick_params(axis='both', labelsize=fontsize-12)
        ax.set_ylabel(column, fontsize=fontsize)
        if n_row == n_rows - 1:
            ax.set_xlabel('time, s', fontsize=fontsize)

        for event_intervals in event_intervals_list:
            # intervals_list = event_intervals.intervals_list
            event_label = event_intervals.label
            color = event_intervals.color
            # mask_interval_list = get_mask_intervals(times, intervals_list=intervals_list)
            mask_interval_list = event_intervals.get_mask_intervals(times)

            for mask_interval in mask_interval_list:
                times_with_mask = times.loc[mask_interval]
                data2plot_with_mask = data2plot.loc[mask_interval]
                ax.plot(
                    times_with_mask,
                    data2plot_with_mask.values,
                    # label=event_label,
                    color=color,
                    alpha=alpha,
                )

            ax.plot([], [], label=event_label, color=color)
        ax.legend(loc='upper right', fontsize=fontsize-8)  # maybe the constant could be generalized

    if plot_suptitle:  # TODO: deal with suptitle
        suptitle = f'{sensor_name.capitalize()} sensors data, session_id = {session_id}'  # TODO: adapt for multiple case
        fig.suptitle(suptitle, fontsize=fontsize + 2)

    fig.tight_layout(rect=[0, 0.00, 1, 0.9625])

    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # fig.subplots_adjust(top=0.5)
    # fig.tight_layout()
    # plt.savefig(pic_prefix + f'measurements_{self.sensor_name}_{self.session_id}.png')
    plt.savefig(pic_path)
    plt.close()


# def plot_measurements_iop(  ### WARNING: CUSTOM FUNCTION FOR IOP-2019
#         analyser_column_pairs_list,  # analysers for hrm, temperature, etc. for the same session_id
#         pic_prefix,
#         session_id,
#         event_intervals_list=None,
#         n_rows=None,
#         n_cols=None,
#         figsize=(21, 15),
#         plot_suptitle=False,
#         fontsize=18,
#         alpha=0.9,
#         alpha_background=0.5,
#         sharex='col',
# ):
#     n_plots = len(analyser_column_pairs_list)
#
#     if n_rows is None:  # TODO: can be isolated to a function
#         if n_cols is None:
#             n_rows, n_cols = get_aspect_from_n_plots(n_plots)
#         else:
#             n_rows = int(np.ceil(n_plots / n_cols))
#     else:
#         if n_cols is None:
#             n_cols = int(np.ceil(n_plots / n_rows))
#
#     analysers_names = [analyser.sensor_name for analyser, column in analyser_column_pairs_list]
#     analysers_names = list(OrderedDict.fromkeys(analysers_names))  # To preserve uniqueness and order
#     pic_path = pic_prefix + 'measurements_' + '_'.join(analysers_names) + f'_{session_id}' + '.png'
#
#     fig, ax_list = plt.subplots(n_rows, n_cols, sharex=sharex, figsize=figsize, squeeze=False)
#     rows_cols_list = itertools.product(range(n_rows), range(n_cols))
#
#     for analyser_column_pair, row_col_pair in zip(analyser_column_pairs_list, rows_cols_list):
#         analyser, column = analyser_column_pair
#         n_row, n_col = row_col_pair
#         ax = ax_list[n_row, n_col]
#
#         times = analyser.df['time']
#         data2plot = analyser.df[column]
#         sensor_name = analyser.sensor_name
#
#         ax.plot(times, data2plot.values, label='nothing', color='black', alpha=alpha_background)
#         ax.tick_params(axis='both', labelsize=fontsize-12)
#         # ax.set_ylabel(column, fontsize=fontsize)
#         ax.set_ylabel(column.split('_')[1], fontsize=fontsize)
#
#         if n_row == n_rows - 1:
#             ax.set_xlabel('time (s)', fontsize=fontsize)
#         elif n_row == 0:
#             sensor_name = 'Accelerometer' if (n_col == 0) else 'Gyroscope'
#             ax.set_title(sensor_name, fontsize=fontsize+3)
#
#         for event_intervals in event_intervals_list:
#             # intervals_list = event_intervals.intervals_list
#             event_label = event_intervals.label
#             color = event_intervals.color
#             # mask_interval_list = get_mask_intervals(times, intervals_list=intervals_list)
#             mask_interval_list = event_intervals.get_mask_intervals(times)
#
#             for mask_interval in mask_interval_list:
#                 times_with_mask = times.loc[mask_interval]
#                 data2plot_with_mask = data2plot.loc[mask_interval]
#                 ax.plot(
#                     times_with_mask,
#                     data2plot_with_mask.values,
#                     # label=event_label,
#                     color=color,
#                     alpha=alpha,
#                 )
#
#             ax.plot([], [], label=event_label, color=color)
#         ax.legend(loc='upper right', fontsize=fontsize-8)  # maybe the constant could be generalized
#
#     if plot_suptitle:  # TODO: deal with suptitle
#         # suptitle = f'{sensor_name.capitalize()} sensors data, session_id = {session_id}'  # TODO: adapt for multiple case
#         suptitle = f'Session id {session_id}'  # TODO: adapt for multiple case
#         fig.suptitle(suptitle, fontsize=fontsize + 3)
#
#     fig.tight_layout(rect=[0, 0.00, 1, 0.9625])
#
#     # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     # fig.subplots_adjust(top=0.5)
#     # fig.tight_layout()
#     # plt.savefig(pic_prefix + f'measurements_{self.sensor_name}_{self.session_id}.png')
#     plt.savefig(pic_path)
#     plt.close()











