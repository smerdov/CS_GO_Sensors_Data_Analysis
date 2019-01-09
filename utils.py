



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






