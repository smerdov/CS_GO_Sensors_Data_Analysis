import json
import pandas as pd
import dateutil.parser

coefs_dict = {
    'gyro_coef': 250.0/32768.0,
    'acc_coef': 2.0/32768.0,
    'mag_coef': 4912.0/32760.0,  # Actually it depends on x, y, z
}

def normalize_MPU9250_data(df, coefs_dict=None):
    df = df.copy()

    if coefs_dict is None:
        coefs_dict = {
            'gyro_coef': 250.0/32768.0,
            'acc_coef': 2.0/32768.0,
            'mag_coef': 4912.0/32760.0,  # Actually it depends on x, y, z
        }

    acc_columns = [column for column in df.columns if column.startswith('acc')]
    gyro_columns = [column for column in df.columns if column.startswith('gyro')]
    mag_columns = [column for column in df.columns if column.startswith('mag')]

    df.loc[:, acc_columns] = df.loc[:, acc_columns] * coefs_dict['acc_coef']
    df.loc[:, gyro_columns] = df.loc[:, gyro_columns] * coefs_dict['gyro_coef']
    df.loc[:, mag_columns] = df.loc[:, mag_columns] * coefs_dict['mag_coef']  # Actually it depends on x, y, z

    return df

def split_df(df, n_chunks, chunk_lenght=100 * 600):
    n_samples = df.shape[0]

    max_possible_chunks = n_samples // chunk_lenght
    # print(max_possible_chunks)
    n_chunks = min(max_possible_chunks, n_chunks)

    if n_chunks < 1:
        return [df.copy()]

    residual_sum = n_samples - n_chunks * chunk_lenght
    residual = residual_sum // (2 * n_chunks)
    # print(n_chunks)
    # print(residual_sum)
    # print(residual)

    chunks_list = []

    for n_chunk in range(n_chunks):
        index_start = residual * (2 * n_chunk + 1) + n_chunk * chunk_lenght
        index_end = residual * (2 * n_chunk + 1) + (n_chunk + 1) * chunk_lenght
        df_chunk = df.iloc[index_start:index_end, :].copy().reset_index(drop=True)
        chunks_list.append(df_chunk)

    return chunks_list


def get_chunks_timestamps(timestamp_min, timestamp_max, chunk_duration, max_chunks):
    timestamp_diff = timestamp_max - timestamp_min
    max_possible_chunks = int(timestamp_diff // chunk_duration)
    # print(max_possible_chunks)
    if max_chunks is not None:
        n_chunks = min(max_possible_chunks, max_chunks)
    else:
        n_chunks = max_possible_chunks

    # print(n_chunks)

    if n_chunks == 0:
        return [timestamp_min, timestamp_max]

    residual_sum = timestamp_diff - n_chunks * chunk_duration
    # residual = residual_sum // (2 * n_chunks)
    residual = residual_sum / (2 * n_chunks)
    # print(residual)

    timestamp_start_end_list = []

    for n_chunk in range(n_chunks):
        timestamp_start = timestamp_min + residual * (2 * n_chunk + 1) + n_chunk * chunk_duration
        timestamp_end = timestamp_min + residual * (2 * n_chunk + 1) + (n_chunk + 1) * chunk_duration
        timestamp_start_end_list.append([timestamp_start, timestamp_end])

    # print(timestamp_start_end_list)
    return timestamp_start_end_list


def split_dfs_by_time(df_list, timestamp_min, timestamp_max, chunk_duration=10 * 60, max_chunks=3, time_col='time'):
    timestamp_start_end_list = get_chunks_timestamps(timestamp_min, timestamp_max, chunk_duration, max_chunks)

    df_chunks_list = []

    for df in df_list:
        timestamp_column = df[time_col]
        # timestamp_column = pd.to_datetime(df[time_col]).apply(lambda x: x.timestamp())
        # print(timestamp_column)
        # timestamp_min = timestamp_column.min()
        # timestamp_max = timestamp_column.max()

        chunks_list = []

        for timestamp_start, timestamp_end in timestamp_start_end_list:
            mask = (timestamp_start <= timestamp_column) & (timestamp_column <= timestamp_end)
            df_chunk = df.loc[mask, :].copy().reset_index(drop=True)
            df_chunk[time_col] = df_chunk[time_col] - timestamp_start # Important
            chunks_list.append(df_chunk)

        df_chunks_list.append(chunks_list)

    return df_chunks_list


def string2json(string):
    string = string.replace("\'", "\"")
    string_json = json.loads(string)

    return string_json

def get_interval_from_moment(moment, interval_start, interval_end):
    return [moment + interval_start, moment + interval_end]

def get_intervals_from_moments(moments, interval_start=-3, interval_end=3):
    intervals = []

    for moment in moments:
        interval = get_interval_from_moment(moment, interval_start=interval_start, interval_end=interval_end)
        intervals.append(interval)

    return intervals

class EventIntervals:

    def __init__(self, intervals_list, label, color):
        self.intervals_list = intervals_list
        self.label = label
        self.color = color

    @staticmethod
    def get_mask_interval(time_column, interval):
        interval_start, interval_end = interval
        mask = (interval_start <= time_column) & (time_column <= interval_end)
        return mask

    def get_mask_intervals(self, time_column):
        # One mask for each interval
        masks_list = []

        for interval in self.intervals_list:
            mask_interval = self.get_mask_interval(time_column, interval)
            masks_list.append(mask_interval)

        return masks_list

def parse_string_iso_format(s):
    d = dateutil.parser.parse(s)
    return d


# def _get_mask_intervals(self, intervals_list):
#     # One mask for all intervals
#     mask = None
#
#     for interval in intervals_list:
#         mask_interval = self._get_mask_interval(interval)
#
#         if mask is None:
#             mask = mask_interval
#         else:
#             mask = mask | mask_interval
#
#     return mask



