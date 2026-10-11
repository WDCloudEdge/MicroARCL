import pytz
from datetime import datetime


def time_string_2_timestamp(time_string):
    # Set the Beijing timezone
    beijing_tz = pytz.timezone('Asia/Shanghai')

    # Convert the time string to a datetime object
    dt_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')

    # Convert the datetime object to Beijing time
    dt_object = beijing_tz.localize(dt_object)

    # Use timestamp() to convert the datetime object to a timestamp
    return int(dt_object.timestamp())


def timestamp_2_time_string(timestamp):
    # Set the Beijing timezone
    beijing_tz = pytz.timezone('Asia/Shanghai')

    # Convert the timestamp to a datetime object
    dt_object = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)

    # Convert the datetime object to Beijing time
    dt_object = dt_object.astimezone(beijing_tz)

    return dt_object.strftime("%Y-%m-%d %H:%M:%S")
