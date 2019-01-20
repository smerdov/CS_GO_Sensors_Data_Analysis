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
        return df

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



