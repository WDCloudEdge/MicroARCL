from sklearn import preprocessing
from datetime import datetime
import pytz
import pandas as pd


def normalize_dataframe(data):
    # Get the DataFrame column names
    data_without_time = data.drop(['timestamp'], axis=1)

    # Normalize each column
    normalized_data = preprocessing.normalize(data_without_time.values, axis=0)

    # Create a new DataFrame using the original column names
    normalized_df = pd.DataFrame(normalized_data, columns=data_without_time.columns)
    normalized_df['timestamp'] = data['timestamp']

    return normalized_df


def df_time_limit(df, begin_timestamp, end_timestamp):
    begin_index = 0
    end_index = 1

    max_timestamp = time_string_2_timestamp(df['timestamp'][df.shape[0] - 1])
    for index, row in df.iterrows():
        if time_string_2_timestamp(row['timestamp']) >= int(begin_timestamp):
            begin_index = index
            break
    for index, row in df.iterrows():
        if index > begin_index and time_string_2_timestamp(row['timestamp']) >= int(end_timestamp):
            end_index = index
            break
    if max_timestamp < int(end_timestamp):
        end_index = df.shape[0] - 1
    if time_string_2_timestamp(df.loc[end_index]['timestamp']) == int(end_timestamp):
        end_index += 1
    df = df.loc[begin_index:end_index]
    df = df.reset_index(drop=True)
    return df


def df_time_limit_normalization(df, begin_timestamp, end_timestamp):
    return normalize_dataframe(df_time_limit(df, begin_timestamp, end_timestamp).fillna(0))


def time_string_2_timestamp(time_string):
    dt_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S').replace(tzinfo=pytz.utc)
    # Use timestamp() to convert the datetime object to a timestamp
    return int(dt_object.timestamp())


class Simple:
    """Descriptor for one labeled fault case: [begin, end] window, label,
    root-cause service, and the sample directory under the dataset root."""
    def __init__(self, global_now_time, global_end_time, label, root_cause, dir):
        self.global_now_time = global_now_time
        self.global_end_time = global_end_time
        self.label = label
        self.root_cause = root_cause
        self.dir = dir
