import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from GeneralAnalyser import GeneralAnalyser

# TODO: add the name because the same analyser can be used for many plots

class ChairAnalyser(GeneralAnalyser):

    def __init__(self,
                 df,
                 pic_prefix,
                 sensor_name,
                 session_id,
                 events_intervals_list=None,
                 interval=2,
                 # measurement_interval=0.01,
                 # measurements_per_batch=1000,
                 name=None,
                 reaction_multiplier=5,
                 ):
        super().__init__(df, pic_prefix, sensor_name, session_id, name)

        self.interval = interval
        self.reaction_multiplier = reaction_multiplier
        self.events_intervals_list = events_intervals_list
        # self.sensor_columns = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        self.sensor_columns = ['acc_x', 'gyro_x', 'acc_y', 'gyro_y', 'acc_z', 'gyro_z']

    def get_floating_features(self, interval=2):
        # time_interval = f'{interval}s'
        time_interval = f'{int(interval * 1000)}ms'

        df2roll = self.df.loc[:, ['time'] + self.sensor_columns].set_index('time')
        df2roll.index = pd.to_timedelta(df2roll.index, unit='s')
        df_rolling = df2roll.rolling(time_interval)  # Can't be centered by default
        stds = df_rolling.std()
        # stds.columns = [f'{column}_std_{time_interval}' for column in stds.columns]
        stds.columns = [f'{column}_std' for column in stds.columns]
        stds.reset_index(drop=True, inplace=True)

        return stds

    def _append_floating_features(self, interval=2):
        floating_std = self.get_floating_features(interval)

        self.df = pd.concat([self.df, floating_std], axis=1)


    def get_reactions_mask(self, floating_std, medians, reaction_multiplier):
        reaction_levels = medians * reaction_multiplier
        reactions_masks = floating_std > reaction_levels.values

        return reactions_masks

    def get_events_masks_dict(self, events_intervals_list):
        events_masks_dict = {}

        for events_intervals in events_intervals_list:
            mask_interval = events_intervals.get_mask_intervals_union(self.df['time'])
            event_label = events_intervals.label
            events_masks_dict[event_label] = mask_interval

        return events_masks_dict

    def get_reaction_events_features(self, reactions_mask, events_masks_dict):
        reactions_mask_sum = reactions_mask.sum(axis=0)

        reaction_events_features_dict = {}

        for event_label, event_mask in events_masks_dict.items():
            event_mask_sum = event_mask.sum()
            # print(type(reactions_mask), type(event_mask))
            reactions_mask_events = reactions_mask.values & event_mask.reshape(-1, 1)
            reactions_mask_events_sum = pd.Series(reactions_mask_events.sum(axis=0), index=reactions_mask_sum.index)

            # events_in_reactions = reactions_mask_events_sum / reactions_mask_sum
            # events_in_reactions.index = [f'events_in_reactions__{event_label}__{index}'  for index in events_in_reactions.index]
            # reaction_events_features_dict.update(events_in_reactions.to_dict())

            reactions_in_events = reactions_mask_events_sum / event_mask_sum
            # reactions_in_events.index = [f'reactions_in_events__{event_label}__{index}'  for index in reactions_in_events.index]
            reactions_in_events.index = [f'moving_{event_label}_{index[:-4]}'  for index in reactions_in_events.index]
            reaction_events_features_dict.update(reactions_in_events.to_dict())

        return reaction_events_features_dict

    def get_reaction_features(self, reactions_mask):
        reactions_mask_mean = reactions_mask.mean(axis=0)
        # reactions_mask_mean.index = [f'reactions_{index}' for index in reactions_mask_mean.index]
        reactions_mask_mean.index = [f'moving_{index[:-4]}' for index in reactions_mask_mean.index]

        return reactions_mask_mean.to_dict()

    def get_lean_back_portion(self, acc_z_threshold=0.97):
        lean_back_portion = (self.df[['acc_z']] < acc_z_threshold).mean()
        # lean_back_portion.index = ['lean_back_portion']
        # lean_back_portion.name = self.name

        return {
            # 'lean_back_portion': lean_back_portion.values[0],
            'lean_back': lean_back_portion.values[0],
        }

    def get_features(self, interval=None, reaction_multiplier=None):
        if interval is None:
            interval = self.interval

        if reaction_multiplier is None:
            reaction_multiplier = self.reaction_multiplier

        floating_std = self.get_floating_features(interval)
        floating_std_median = floating_std.quantile(0.5, axis=0)
        # floating_std_median.index = [f'median_{index}' for index in floating_std_median.index]
        floating_std_median.index = [f'med_{index}' for index in floating_std_median.index]

        reactions_mask = self.get_reactions_mask(floating_std, floating_std_median, reaction_multiplier=reaction_multiplier)

        reaction_features = self.get_reaction_features(reactions_mask)
        oscillations_features = floating_std_median.to_dict()
        lean_back_portion = self.get_lean_back_portion()

        features_list = [reaction_features, oscillations_features, lean_back_portion]

        if self.events_intervals_list is not None:
            events_masks_dict = self.get_events_masks_dict(self.events_intervals_list)
            reaction_events_features = self.get_reaction_events_features(reactions_mask, events_masks_dict)
            features_list.append(reaction_events_features)

        all_features_dict = {}
        for features in features_list:
            all_features_dict.update(features)

        all_features = pd.Series(all_features_dict, name=self.session_id)

        return all_features

    # # def create_mean_stds(self, columns=('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')):
    # def create_mean_stds(self):
    #     df_chair = self.df.loc[:, self.sensor_columns]
    #     # df_chair = df_chair.loc[:, columns]
    #     # medians, lower_bounds, upper_bounds = np.percentile(df_chair, [50, percentile2crop, 100 - percentile2crop], axis=0)
    #
    #     means = df_chair.mean(axis=0)
    #     medians = df_chair.median(axis=0)
    #     stds = df_chair.std(axis=0)
    #
    #     return means, stds, medians
    #
    # def get_nonstationary_values_portion(self, n_sigma=3):
    #     means = self.means
    #     stds = self.stds
    #
    #     columns = stds.index
    #     df_chair = self.df.loc[:, columns]
    #
    #     lower_bounds = means - n_sigma * stds
    #     upper_bounds = means + n_sigma * stds
    #
    #     low_values_means = (df_chair.loc[:, columns] < lower_bounds).mean()
    #     high_values_means = (df_chair.loc[:, columns] > upper_bounds).mean()
    #
    #     nonstationary_values_portion = low_values_means + high_values_means
    #     nonstationary_values_portion.index = [colname + '__nonstationary_portion' for colname in nonstationary_values_portion.index]
    #     nonstationary_values_portion.name = self.name
    #
    #     return nonstationary_values_portion
    #
    # def get_oscillation_intensity(self, percentile2crop=10, columns=('acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z')):
    #     df_chair = self.df.loc[:, columns]
    #     result = {}
    #
    #     for column in columns:
    #         lower_bounds, upper_bounds = np.percentile(df_chair.loc[:, column], [percentile2crop, 100 - percentile2crop], axis=0)
    #         # intervals = upper_bounds - lower_bounds
    #         low_values_mask = (df_chair.loc[:, column] < lower_bounds)
    #         high_values_mask = (df_chair.loc[:, column] > upper_bounds)
    #
    #         normal_values_mask = (~low_values_mask) & (~high_values_mask)
    #
    #         usual_sitting_stds = df_chair.loc[normal_values_mask, column].std()
    #         oscillations = usual_sitting_stds# / intervals
    #         feature_name = f'{column}__oscillations'
    #         result[feature_name] = oscillations
    #
    #     result = pd.Series(result)
    #     result.name = self.name
    #
    #     return result







